import json
import logging
import time
import os
import io
import re
import numpy as np
import enchant
import matplotlib.pyplot as plt
import multiprocessing
import concurrent
import asyncio
import matplotlib
from matplotlib import font_manager
from dotenv import load_dotenv
from thefuzz import fuzz, process
from typing import List, Dict, Tuple
from google.cloud import vision
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from wordcloud import WordCloud
from collections import defaultdict, Counter

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the 'Agg' backend which doesn't require a GUI
matplotlib.use('Agg')

MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))

# Simple in-memory brand database
BRAND_DATABASE = {
    "pepsi": {"variations": ["pepsi cola", "pepsi-cola"], "category": "beverage"},
    "coca cola": {"variations": ["coke", "coca-cola"], "category": "beverage"},
    "pizza hut": {"variations": ["pizzahut"], "category": "food"},
    "doritos": {"variations": ["dorito"], "category": "food"},
    "reebok": {"variations": ["rebok"], "category": "apparel"},
}

d = enchant.Dict("en_US")

def load_ocr_results(bucket: storage.Bucket, video_id: str) -> List[Dict]:
    ocr_blob = bucket.blob(f'{video_id}/ocr/ocr_results.json')
    if ocr_blob.exists():
        return json.loads(ocr_blob.download_as_string())
    else:
        raise FileNotFoundError(f"OCR results not found for video: {video_id}")

async def save_processed_ocr_results(bucket: storage.Bucket, video_id: str, cleaned_results: List[Dict]):
    processed_ocr_blob = bucket.blob(f'{video_id}/ocr/processed_ocr.json')
    
    # Use asyncio.to_thread to run the synchronous upload in a separate thread
    await asyncio.to_thread(
        processed_ocr_blob.upload_from_string,
        json.dumps(cleaned_results, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"Saved processed OCR results for video: {video_id}")

async def save_brands_ocr_results(bucket: storage.Bucket, video_id: str, brand_results: List[Dict]):
    brands_ocr_blob = bucket.blob(f'{video_id}/ocr/brands_ocr.json')
    
    await asyncio.to_thread(
        brands_ocr_blob.upload_from_string,
        json.dumps(brand_results, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"Saved brands OCR results for video: {video_id}")

async def create_and_save_brand_table(bucket: storage.Bucket, video_id: str, brand_results: List[Dict], fps: float):
    brand_stats = defaultdict(lambda: {"frame_count": 0, "time_on_screen": 0})

    for frame in brand_results:
        for brand in frame['detected_brands']:
            brand_name = brand['text']
            brand_stats[brand_name]['frame_count'] += 1

    for brand, stats in brand_stats.items():
        stats['time_on_screen'] = round(stats['frame_count'] / fps, 2)

    brand_table = dict(brand_stats)

    brand_table_blob = bucket.blob(f'{video_id}/ocr/brands_table.json')
    
    await asyncio.to_thread(
        brand_table_blob.upload_from_string,
        json.dumps(brand_table, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"Brand table created and saved for video: {video_id}")

def clean_ocr_data(frame: Dict) -> Dict:
    """
    Clean up OCR data in a single frame:
    - Remove single letters/numbers
    - Remove partial words
    - Correct misspelled words
    """
    cleaned_annotations = []
    for annotation in frame.get('text_annotations', []):
        text = annotation['text']
        
        # Skip single letters/numbers
        if len(text) <= 1:
            continue
        
        # Remove non-alphanumeric characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Skip if the cleaned text is empty or too short
        if len(cleaned_text) <= 2:
            continue
        
        # Check if it's a valid word or correct it
        words = cleaned_text.split()
        corrected_words = []
        for word in words:
            if d.check(word):
                corrected_words.append(word)
            else:
                suggestions = d.suggest(word)
                if suggestions:
                    corrected_words.append(suggestions[0])
        
        # Skip if no valid words remain
        if not corrected_words:
            continue
        
        # Join the corrected words back into a string
        final_text = ' '.join(corrected_words)
        
        cleaned_annotations.append({
            "text": final_text,
            "original_text": text,
            "bounding_box": annotation['bounding_box']
        })
    
    return {
        "frame_number": frame['frame_number'],
        "full_text": ' '.join([ann['text'] for ann in cleaned_annotations]),
        "cleaned_annotations": cleaned_annotations
    }

def process_single_frame(frame: Dict) -> Dict:
    cleaned_frame = clean_ocr_data(frame)
    return consolidate_words(cleaned_frame)

def clean_and_consolidate_ocr_data(ocr_results: List[Dict]) -> List[Dict]:
    total_frames = len(ocr_results)
    cleaned_results = []
    
    # Determine the number of workers (use CPU count - 1 to leave one core free)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to process frames in parallel
        futures = list(executor.map(process_single_frame, ocr_results))
        
        # Collect results and log progress
        for i, result in enumerate(futures):
            cleaned_results.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == total_frames:
                logger.info(f"Cleaned and consolidated {i + 1}/{total_frames} frames")
    
    return cleaned_results

def fuzzy_match_brand(text: str, min_score: int = 95) -> Tuple[str, int]:
    try:
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()
        if len(cleaned_text) < 2:
            return None, 0
        best_match, score = process.extractOne(cleaned_text, BRAND_DATABASE.keys())
        return (best_match, score) if score >= min_score else (None, 0)
    except Exception as e:
        logger.error(f"Error in fuzzy_match_brand: {str(e)}")
        return None, 0

def are_words_close(word1: Dict, word2: Dict, proximity_threshold: int = 50) -> bool:
    """
    Check if two words are close enough to be considered part of the same brand name.
    Proximity is measured as the distance between the bounding boxes.
    """
    box1 = word1['bounding_box']
    box2 = word2['bounding_box']

    # Calculate the horizontal and vertical distances between the two bounding boxes
    hor_distance = abs(box1['vertices'][1]['x'] - box2['vertices'][0]['x'])
    vert_distance = abs(box1['vertices'][0]['y'] - box2['vertices'][0]['y'])

    return hor_distance < proximity_threshold and vert_distance < proximity_threshold

def merge_bounding_boxes(box1: Dict, box2: Dict) -> Dict:
    """
    Merge two bounding boxes into one that encloses both.
    """
    all_vertices = box1['vertices'] + box2['vertices']
    min_x = min(vertex['x'] for vertex in all_vertices)
    max_x = max(vertex['x'] for vertex in all_vertices)
    min_y = min(vertex['y'] for vertex in all_vertices)
    max_y = max(vertex['y'] for vertex in all_vertices)
    
    return {
        "vertices": [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y}
        ]
    }

def consolidate_words(frame: Dict) -> Dict:
    annotations = frame['cleaned_annotations']
    consolidated = []
    i = 0
    while i < len(annotations):
        current_word = annotations[i]
        if i + 1 < len(annotations):
            next_word = annotations[i + 1]
            if are_words_close(current_word, next_word):
                combined_text = f"{current_word['text']} {next_word['text']}"
                match, score = fuzzy_match_brand(combined_text)
                if match:
                    consolidated.append({
                        "text": match,
                        "original_text": combined_text,
                        "confidence": score,
                        "bounding_box": merge_bounding_boxes(current_word['bounding_box'], next_word['bounding_box'])
                    })
                    i += 2
                    continue
        match, score = fuzzy_match_brand(current_word['text'])
        if match:
            current_word['text'] = match
            current_word['confidence'] = score
        consolidated.append(current_word)
        i += 1
    frame['cleaned_annotations'] = consolidated
    return frame

def interpolate_brand(current_brand: Dict, next_brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    t = (frame_number - current_frame) / (next_frame - current_frame)
    interpolated_box = {
        'vertices': [
            {k: int(current_brand['bounding_box']['vertices'][j][k] + 
                     t * (next_brand['bounding_box']['vertices'][j][k] - 
                          current_brand['bounding_box']['vertices'][j][k]))
             for k in ['x', 'y']}
            for j in range(4)
        ]
    }
    confidence_decay = np.exp(-0.5 * (frame_number - current_frame) / fps)
    interpolated_confidence = current_brand['confidence'] * confidence_decay

    return {
        "text": current_brand['text'],
        "original_text": current_brand['text'],
        "confidence": interpolated_confidence,
        "bounding_box": interpolated_box,
        "is_interpolated": True
    }

def detect_brands_and_interpolate(cleaned_results: List[Dict], fps: float) -> List[Dict]:
    brand_results = []
    brand_appearances = defaultdict(list)

    # First pass: Detect brands
    for frame in cleaned_results:
        frame_number = frame['frame_number']
        detected_brands = []
        for annotation in frame['cleaned_annotations']:
            if 'confidence' in annotation:
                detected_brands.append(annotation)
                brand_appearances[annotation['text']].append((frame_number, annotation))
        
        brand_results.append({
            "frame_number": frame_number,
            "detected_brands": detected_brands
        })

    # Second pass: Interpolate brands
    interpolation_window = int(fps)  # Set to approximately one second
    for brand, appearances in brand_appearances.items():
        for i in range(len(appearances) - 1):
            current_frame, current_brand = appearances[i]
            next_frame, next_brand = appearances[i + 1]

            for frame_number in range(current_frame + 1, next_frame):
                if frame_number - current_frame <= interpolation_window:
                    interpolated_brand = interpolate_brand(current_brand, next_brand, current_frame, next_frame, frame_number, fps)
                    frame_index = next((i for i, f in enumerate(brand_results) if f['frame_number'] == frame_number), None)
                    if frame_index is not None:
                        brand_results[frame_index]['detected_brands'].append(interpolated_brand)

    # Sort detected brands in each frame by confidence
    for frame in brand_results:
        frame['detected_brands'].sort(key=lambda x: x['confidence'], reverse=True)

    return brand_results

def find_font(font_names):
    """
    Find the first available font from the given list of font names.
    """
    for font_name in font_names:
        try:
            return font_manager.findfont(font_manager.FontProperties(family=font_name))
        except:
            continue
    return None

def create_word_cloud(bucket: storage.Bucket, video_id: str, cleaned_results: List[Dict]):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font.
    """
    # Extract all text annotations
    all_text = []
    for frame in cleaned_results:
        for annotation in frame['cleaned_annotations']:
            all_text.append(annotation['text'].lower())

    # Count word frequencies
    word_freq = Counter(all_text)

    # Create a mask image (circle shape)
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # Define font preference order
    preferred_fonts = [
        'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans',
        'Roboto', 'Open Sans', 'Lato', 'Noto Sans'
    ]

    # Find the first available preferred font
    font_path = find_font(preferred_fonts)
    if not font_path:
        logger.warning("No preferred font found. Using default.")
    else:
        logger.info(f"Using font: {font_path}")

    # Generate word cloud
    wordcloud = WordCloud(width=600, height=600,
                          background_color='white',
                          max_words=100,  # Limit to top 100 words for clarity
                          min_font_size=10,
                          max_font_size=120,
                          mask=mask,
                          font_path=font_path,
                          colormap='ocean',  # Use a colorful colormap
                          prefer_horizontal=0.9,
                          scale=2
                         ).generate_from_frequencies(word_freq)

    # Create a figure and save it to a BytesIO object
    plt.figure(figsize=(10,10), frameon=False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    img_buffer.seek(0)

    # Upload to bucket
    wordcloud_blob = bucket.blob(f'{video_id}/ocr/wordcloud.jpg')
    wordcloud_blob.upload_from_file(img_buffer, content_type='image/jpg')
    logger.info(f"Word cloud created and saved for video: {video_id}")

    plt.close()

async def process_single_frame_ocr(frame_blob, video_id, frame_number):
    try:
        # Create a client
        client = vision.ImageAnnotatorClient()

        # Read the image file
        image_content = frame_blob.download_as_bytes()

        # Create an Image object
        image = vision.Image(content=image_content)

        # Perform OCR on the image
        response = client.text_detection(image=image)

        # Process the response
        texts = response.text_annotations

        if texts:
            # The first annotation contains the entire detected text
            full_text = texts[0].description

            # Process individual text annotations
            text_annotations = []
            for annotation in texts[1:]:  # Skip the first one as it's the full text
                text_annotation = {
                    "text": annotation.description,
                    "bounding_box": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in annotation.bounding_poly.vertices
                        ]
                    }
                }
                text_annotations.append(text_annotation)

            return {
                "frame_number": frame_number,
                "full_text": full_text,
                "text_annotations": text_annotations
            }
        else:
            logger.info(f"No text found in frame {frame_number}")
            return {
                "frame_number": frame_number,
                "full_text": "",
                "text_annotations": []
            }

    except Exception as e:
        logger.error(f"Error processing OCR for frame {frame_number}: {str(e)}")
        return None

async def post_process_ocr(video_id: str, fps: float, bucket: storage.Bucket):
    try:
        # Load OCR results
        ocr_results = load_ocr_results(bucket, video_id)
        
        # Step 1: Clean and consolidate OCR data
        logger.info(f"Cleaning and consolidating OCR data for video: {video_id}")
        cleaned_results = clean_and_consolidate_ocr_data(ocr_results)
        await save_processed_ocr_results(bucket, video_id, cleaned_results)
        
        # Step 2: Create word cloud
        logger.info(f"Creating word cloud for video: {video_id}")
        create_word_cloud(bucket, video_id, cleaned_results)
        
        # Step 3: Detect brands and interpolate
        logger.info(f"Detecting brands and interpolating for video: {video_id}")
        brand_results = detect_brands_and_interpolate(cleaned_results, fps)
        
        # Step 4: Save brands OCR results
        await save_brands_ocr_results(bucket, video_id, brand_results)
        
        # Step 5: Create and save brand table
        await create_and_save_brand_table(bucket, video_id, brand_results, fps)
        
        logger.info(f"Completed post-processing OCR for video: {video_id}")
        return brand_results
    except Exception as e:
        logger.error(f"Error in post_process_ocr for video {video_id}: {str(e)}")
        raise

async def process_ocr(video_id: str, bucket: storage.Bucket, status_tracker: 'StatusTracker'):
    logger.info(f"Starting OCR processing for video: {video_id}")
    status_tracker.update_process_status("ocr", "in_progress", 0)

    ocr_start_time = time.time()
    frame_blobs = list(bucket.list_blobs(prefix=f'{video_id}/frames/'))
    total_frames = len(frame_blobs)

    ocr_results = []
    processed_frames = 0
    total_words = 0

    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all frame processing tasks
        future_to_frame = {executor.submit(process_single_frame_ocr, frame_blob, video_id, int(frame_blob.name.split('/')[-1].split('.')[0])): frame_blob for frame_blob in frame_blobs}

        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_frame):
            try:
                result = future.result()
                if result:
                    ocr_results.append(result)
                    total_words += len(result['full_text'].split())

                processed_frames += 1
                progress = (processed_frames / total_frames) * 100
                status_tracker.update_process_status("ocr", "in_progress", progress)

                # Log progress periodically
                if processed_frames % 100 == 0 or processed_frames == total_frames:
                    logger.info(f"Processed {processed_frames}/{total_frames} frames for video {video_id}")

            except Exception as exc:
                logger.error(f"Error processing frame for video {video_id}: {str(exc)}")

    # Sort OCR results by frame number
    ocr_results.sort(key=lambda x: x['frame_number'])

    # Store OCR results
    ocr_blob = bucket.blob(f'{video_id}/ocr/ocr_results.json')
    await asyncio.to_thread(ocr_blob.upload_from_string, json.dumps(ocr_results, indent=2), content_type='application/json')

    ocr_processing_time = time.time() - ocr_start_time
    frames_with_text = len([frame for frame in ocr_results if frame['text_annotations']])

    ocr_stats = {
        "ocr_processing_time": f"{ocr_processing_time:.2f} seconds",
        "frames_processed": total_frames,
        "frames_with_text": frames_with_text,
        "total_words_detected": total_words
    }

    status_tracker.update_process_status("ocr", "complete", 100)
    logger.info(f"Completed OCR processing for video: {video_id}")
    return ocr_stats