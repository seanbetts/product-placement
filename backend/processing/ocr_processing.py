import json
import logging
import time
import os
import numpy as np
from dotenv import load_dotenv
from thefuzz import fuzz, process
from typing import List, Dict, Tuple
from google.cloud import vision
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

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
    best_match, score = process.extractOne(text, BRAND_DATABASE.keys())
    if score >= min_score:
        return best_match, score
    return None, 0

def consolidate_words(words: List[Dict]) -> List[Dict]:
    """
    Consolidate nearby words that might form a multi-word brand name.
    """
    consolidated = []
    i = 0
    while i < len(words):
        current_word = words[i]
        if i + 1 < len(words):
            next_word = words[i + 1]
            combined_text = f"{current_word['text']} {next_word['text']}"
            match, score = fuzzy_match_brand(combined_text)
            if match:
                # Combine the words
                consolidated.append({
                    "text": match,
                    "original_text": combined_text,
                    "confidence": score,
                    "bounding_box": {
                        "vertices": current_word['bounding_box']['vertices'] + next_word['bounding_box']['vertices']
                    }
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
            text = annotation['text'].lower()

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

def save_processed_ocr_results(bucket: storage.Bucket, video_id: str, processed_results: List[Dict]):
    """
    Save the processed OCR results to a new file in the storage bucket.
    """
    processed_ocr_blob = bucket.blob(f'{video_id}/processed_ocr.json')
    processed_ocr_blob.upload_from_string(
        json.dumps(processed_results, indent=2),
        content_type='application/json'
    )
    logger.info(f"Saved processed OCR results for video: {video_id}")

async def process_and_save_ocr(video_id: str, fps: int, bucket: storage.Bucket):
    """
    Process the OCR results and save both raw and processed results.
    """
    ocr_blob = bucket.blob(f'{video_id}/ocr_results.json')
    
    if ocr_blob.exists():
        ocr_results = json.loads(ocr_blob.download_as_string())
        processed_results = post_process_ocr(ocr_results, fps)
        
        # Save processed results
        save_processed_ocr_results(bucket, video_id, processed_results)
        
        return processed_results
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
    ocr_blob = bucket.blob(f'{video_id}/ocr_results.json')
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