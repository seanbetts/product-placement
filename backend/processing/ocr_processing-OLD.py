import json
import logging
import time
import os
import re
import numpy as np
import enchant
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib import font_manager
from dotenv import load_dotenv
from thefuzz import fuzz, process
from typing import List, Dict, Tuple
from google.cloud import vision
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from wordcloud import WordCloud
from collections import defaultdict, Counter

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))

# Simple in-memory brand database
BRAND_DATABASE = {
    "pepsi": {"variations": ["pepsi cola", "pepsi-cola"], "category": "beverage"},
    "coca cola": {"variations": ["coke", "coca-cola"], "category": "beverage"},
    "pizza hut": {"variations": ["pizzahut"], "category": "food"},
    "doritos": {"variations": ["dorito"], "category": "food"},
    "reebok": {"variations": ["rebok"], "category": "apparel"},
}

def fuzzy_match_brand(text: str, min_score: int = 95) -> Tuple[str, int]:
    """
    Fuzzy match the given text against the brand database.
    Returns the best match and its score if it's above the minimum score.
    """
    # Remove any non-alphanumeric characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()
    
    # If the cleaned text is empty or too short, return None
    if len(cleaned_text) < 2:
        return None, 0

    best_match, score = process.extractOne(cleaned_text, BRAND_DATABASE.keys())
    if score >= min_score:
        return best_match, score
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

def consolidate_words(words: List[Dict]) -> List[Dict]:
    """
    Consolidate nearby words that might form a multi-word brand name.
    """
    # Sort words by their top-left x coordinate (assuming left-to-right language)
    words.sort(key=lambda word: word['bounding_box']['vertices'][0]['x'])
    consolidated = []
    i = 0

    while i < len(words):
        current_word = words[i]
        if i + 1 < len(words):
            next_word = words[i + 1]
            
            if are_words_close(current_word, next_word):
                combined_text = f"{current_word['text']} {next_word['text']}"
                match, score = fuzzy_match_brand(combined_text)
                
                if match:
                    # Combine words based on proximity and matching
                    bounding_box = merge_bounding_boxes(current_word['bounding_box'], next_word['bounding_box'])
                    consolidated.append({
                        "text": match,
                        "original_text": combined_text,
                        "confidence": score,
                        "bounding_box": bounding_box
                    })
                    i += 2
                    continue

        match, score = fuzzy_match_brand(current_word['text'])
        if match:
            current_word['text'] = match
            current_word['confidence'] = score
        consolidated.append(current_word)
        i += 1

    return consolidated

def interpolate_bounding_box(prev_box, next_box, current_frame, prev_frame, next_frame):
    if not prev_box or not next_box:
        return None
    t = (current_frame - prev_frame) / (next_frame - prev_frame)
    return {
        'vertices': [
            {k: int(prev_box['vertices'][i][k] + t * (next_box['vertices'][i][k] - prev_box['vertices'][i][k]))
             for k in ['x', 'y']}
            for i in range(4)
        ]
    }

# Initialize the spell checker
d = enchant.Dict("en_US")

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

def create_cleaned_ocr_data(ocr_results: List[Dict]) -> List[Dict]:
    """
    Create cleaned OCR data for all frames.
    """
    return [clean_ocr_data(frame) for frame in ocr_results]

def save_cleaned_ocr_data(bucket: storage.Bucket, video_id: str, cleaned_results: List[Dict]):
    """
    Save the cleaned OCR results to a new file in the storage bucket.
    """
    cleaned_ocr_blob = bucket.blob(f'{video_id}/ocr/processed_ocr.json')
    cleaned_ocr_blob.upload_from_string(
        json.dumps(cleaned_results, indent=2),
        content_type='application/json'
    )
    logger.info(f"Saved cleaned OCR results for video: {video_id}")

def post_process_ocr(ocr_output: List[Dict], fps: int) -> List[Dict]:
    processed_output = []
    known_watermarks = ["MOVIECLIPS.COM"]
    brand_appearances = {}
    interpolation_window = fps  # Set to approximately one second

    # First pass: Detect brands and mark frames for interpolation
    for frame in ocr_output:
        frame_number = frame['frame_number']
        detected_brands = []

        for annotation in frame.get('text_annotations', []):
            text = annotation['text']

            # Skip processing if the text is too short or only punctuation
            if len(text) < 2 or not re.search(r'[a-zA-Z0-9]', text):
                continue

            if text.upper() in known_watermarks:
                continue

            match, score = fuzzy_match_brand(text)
            if match and score >= 90:
                detected_brands.append({
                    "text": match,
                    "original_text": text,
                    "confidence": score,
                    "bounding_box": annotation['bounding_box'],
                    "is_interpolated": False
                })

        detected_brands = consolidate_words(detected_brands)

        if detected_brands:
            top_brand = max(detected_brands, key=lambda x: x['confidence'])
            if top_brand['text'] not in brand_appearances:
                brand_appearances[top_brand['text']] = []
            brand_appearances[top_brand['text']].append((frame_number, top_brand))
        
        processed_output.append({
            "frame_number": frame_number,
            "detected_brands": detected_brands if detected_brands else []
        })

    # Second pass: Interpolate bounding boxes
    for brand, appearances in brand_appearances.items():
        for i in range(len(appearances) - 1):
            current_frame, current_brand = appearances[i]
            next_frame, next_brand = appearances[i + 1]

            for frame_number in range(current_frame + 1, next_frame):
                if frame_number - current_frame <= interpolation_window:
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

                    interpolated_brand = {
                        "text": brand,
                        "original_text": brand,
                        "confidence": interpolated_confidence,
                        "bounding_box": interpolated_box,
                        "is_interpolated": True
                    }

                    frame_index = next((i for i, f in enumerate(processed_output) if f['frame_number'] == frame_number), None)
                    if frame_index is not None:
                        processed_output[frame_index]['detected_brands'].append(interpolated_brand)

    # Sort detected brands in each frame by confidence
    for frame in processed_output:
        frame['detected_brands'].sort(key=lambda x: x['confidence'], reverse=True)

    return processed_output

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

def create_word_cloud(bucket: storage.Bucket, video_id: str):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font.
    """
    processed_ocr_blob = bucket.blob(f'{video_id}/ocr/processed_ocr.json')
    if not processed_ocr_blob.exists():
        logger.error(f"Processed OCR results not found for video: {video_id}")
        return

    processed_ocr_data = json.loads(processed_ocr_blob.download_as_string())

    # Extract all text annotations
    all_text = []
    for frame in processed_ocr_data:
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

    plt.figure(figsize=(10,10), frameon=False)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Save to a temporary file
    temp_file = '/tmp/wordcloud.jpg'
    plt.savefig(temp_file, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Upload to bucket
    wordcloud_blob = bucket.blob(f'{video_id}/ocr/wordcloud.jpg')
    wordcloud_blob.upload_from_filename(temp_file, content_type='image/jpeg')
    logger.info(f"Word cloud created and saved for video: {video_id}")

def create_brand_table(bucket: storage.Bucket, video_id: str, fps: float):
    """
    Create a brand table from the brands OCR results and save it as a JSON file.
    """
    brands_ocr_blob = bucket.blob(f'{video_id}/ocr/brands_ocr.json')
    if not brands_ocr_blob.exists():
        logger.error(f"Brands OCR results not found for video: {video_id}")
        return

    brands_ocr_data = json.loads(brands_ocr_blob.download_as_string())
    
    brand_stats = defaultdict(lambda: {"frame_count": 0, "time_on_screen": 0})

    for frame in brands_ocr_data:
        for brand in frame['detected_brands']:
            brand_name = brand['text']
            brand_stats[brand_name]['frame_count'] += 1

    # Calculate time on screen
    for brand, stats in brand_stats.items():
        stats['time_on_screen'] = round(stats['frame_count'] / fps, 2)

    # Convert to regular dict for JSON serialization
    brand_table = dict(brand_stats)

    # Save to JSON file
    brand_table_blob = bucket.blob(f'{video_id}/ocr/brands_table.json')
    brand_table_blob.upload_from_string(
        json.dumps(brand_table, indent=2),
        content_type='application/json'
    )

    logger.info(f"Brand table created and saved for video: {video_id}")

def save_processed_ocr_results(bucket: storage.Bucket, video_id: str, processed_results: List[Dict]):
    """
    Save the processed OCR results to a new file in the storage bucket.
    """
    processed_ocr_blob = bucket.blob(f'{video_id}/ocr/brands_ocr.json')
    processed_ocr_blob.upload_from_string(
        json.dumps(processed_results, indent=2),
        content_type='application/json'
    )
    logger.info(f"Saved processed OCR results for video: {video_id}")

async def process_and_save_ocr(video_id: str, fps: float, bucket: storage.Bucket):
    """
    Process the OCR results, save cleaned results, perform brand detection and interpolation,
    create a word cloud, and generate a brand table.
    """
    ocr_blob = bucket.blob(f'{video_id}/ocr/ocr_results.json')

    if ocr_blob.exists():
        ocr_results = json.loads(ocr_blob.download_as_string())

        # Create and save cleaned OCR results
        cleaned_ocr_results = create_cleaned_ocr_data(ocr_results)
        save_cleaned_ocr_data(bucket, video_id, cleaned_ocr_results)

        # Perform brand detection and interpolation
        final_processed_results = post_process_ocr(ocr_results, fps)

        # Save final processed results
        save_processed_ocr_results(bucket, video_id, final_processed_results)

        # Create word cloud
        create_word_cloud(bucket, video_id)

        # Create brand table
        create_brand_table(bucket, video_id, fps)

        return final_processed_results
    else:
        logger.error(f"OCR results not found for video: {video_id}")
        return None

def process_single_frame_ocr(frame_blob, video_id, frame_number):
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

async def process_ocr(video_id: str, bucket: 'storage.Bucket', status_tracker: 'StatusTracker'):
    logger.info(f"Starting OCR processing for video: {video_id}")
    status_tracker.update_process_status("ocr", "in_progress", 0)

    ocr_start_time = time.time()
    frame_blobs = list(bucket.list_blobs(prefix=f'{video_id}/frames/'))
    total_frames = len(frame_blobs)

    ocr_results = []
    processed_frames = 0
    total_words = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for frame_blob in frame_blobs:
            frame_number = int(frame_blob.name.split('/')[-1].split('.')[0])
            future = executor.submit(process_single_frame_ocr, frame_blob, video_id, frame_number)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                ocr_results.append(result)
                # Count words in the full text of this frame
                total_words += len(result['full_text'].split())

            processed_frames += 1
            progress = (processed_frames / total_frames) * 100
            status_tracker.update_process_status("ocr", "in_progress", progress)

    # Sort OCR results by frame number
    ocr_results.sort(key=lambda x: x['frame_number'])

    # Store OCR results
    ocr_blob = bucket.blob(f'{video_id}/ocr/ocr_results.json')
    ocr_blob.upload_from_string(json.dumps(ocr_results, indent=2), content_type='application/json')

    ocr_processing_time = time.time() - ocr_start_time
    frames_with_text = len([frame for frame in ocr_results if frame['text_annotations']])

    ocr_stats = {
        "ocr_processing_time": f"{ocr_processing_time:.2f} seconds",
        "frames_processed": total_frames,
        "frames_with_text": frames_with_text,
        "total_words_detected": total_words
    }

    status_tracker.update_process_status("ocr", "complete", 100)

    return ocr_stats