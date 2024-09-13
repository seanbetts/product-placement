import json
import logging
import time
import os
import io
import re
import numpy as np
import enchant
import matplotlib.pyplot as plt
import asyncio
import matplotlib
import boto3
from botocore.exceptions import ClientError
from matplotlib import font_manager
from dotenv import load_dotenv
from thefuzz import fuzz, process
from typing import List, Dict, Tuple, Optional, Set, Callable, TYPE_CHECKING
from google.cloud import vision
from wordcloud import WordCloud
from collections import defaultdict, Counter

if TYPE_CHECKING:
    from main import StatusTracker

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the 'Agg' backend which doesn't require a GUI
matplotlib.use('Agg')

MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))
PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')

# Initialize S3 client
s3_client = boto3.client('s3')

# Simple in-memory brand database
BRAND_DATABASE = {
    "pepsi": {"variations": ["pepsi cola", "pepsi-cola"], "category": "beverage"},
    "coca cola": {"variations": ["coke", "coca-cola"], "category": "beverage"},
    "pizza hut": {"variations": ["pizzahut"], "category": "food"},
    "doritos": {"variations": ["dorito"], "category": "food"},
    "reebok": {"variations": ["rebok"], "category": "apparel"},
    "adidas": {"variations": ["addidas", "adiddas", "addiddas"], "category": "apparel"},
    "betano": {"variations": ["beteno"], "category": "gaming"},
    "₿c.game": {"variations": ["bc.game", "☐c.game", "₿c game", "☐c game", "bc game", "₿c-game", "☐c-game", "bc-game", "₿cgame", "bcgame", "bccane", "bc.came", "bc came", "bc cane"], "category": "gaming"},
    "chang": {"variations": ["chan", "ching"], "category": "beverages"},
    "guiness": {"variations": ["guinness", "guinness", "guinnes"], "category": "beverages"},
    "gambleaware": {"variations": ["gamble aware", "gambl aware"], "category": "gaming"},
    "king power": {"variations": ["kingpower", "kingpow"], "category": "travel"},
    "litefinance": {"variations": ["lite finance", "litefinans", "lite finans"], "category": "finance"},
}

d = enchant.Dict("en_US")

def load_ocr_results(s3_client, video_id: str) -> List[Dict]:
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=f'{video_id}/ocr/ocr_results.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise FileNotFoundError(f"OCR results not found for video: {video_id}")
        else:
            raise

async def save_processed_ocr_results(s3_client, video_id: str, cleaned_results: List[Dict]):
    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/processed_ocr.json',
            Body=json.dumps(cleaned_results, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Saved processed OCR results for video: {video_id}")
    except ClientError as e:
        logger.error(f"Error saving processed OCR results for video {video_id}: {str(e)}")
        raise

async def save_brands_ocr_results(s3_client, video_id: str, brand_results: List[Dict]):
    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/brands_ocr.json',
            Body=json.dumps(brand_results, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Saved brands OCR results for video: {video_id}")
    except ClientError as e:
        logger.error(f"Error saving brands OCR results for video {video_id}: {str(e)}")
        raise

def create_and_save_brand_table(s3_client, video_id: str, brand_appearances: Dict[str, Set[int]], fps: float):
    brand_stats = {}
    min_frames = int(fps)  # Minimum number of frames (1 second)

    for brand, frames in brand_appearances.items():
        frame_list = sorted(frames)
        if len(frame_list) >= min_frames:
            brand_stats[brand] = {
                "frame_count": len(frame_list),
                "time_on_screen": round(len(frame_list) / fps, 2),
                "first_appearance": frame_list[0],
                "last_appearance": frame_list[-1]
            }
        else:
            logger.info(f"Discarded brand '{brand}' as it appeared for less than 1 second ({len(frame_list)} frames)")

    try:
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/brands_table.json',
            Body=json.dumps(brand_stats, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Brand table created and saved for video: {video_id}")
        return brand_stats
    except ClientError as e:
        logger.error(f"Error saving brand table for video {video_id}: {str(e)}")
        raise

def clean_ocr_data(frame: Dict, preprocess_func: Callable[[str], str]) -> Dict:
    """
    Clean up OCR data in a single frame:
    - Remove single letters/numbers
    - Remove partial words
    - Correct misspelled words
    - Preprocess OCR text to correct common errors
    """
    cleaned_annotations = []
    for annotation in frame.get('text_annotations', []):
        text = preprocess_func(annotation['text'])
        
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
            "original_text": annotation['text'],
            "bounding_box": annotation['bounding_box']
        })
    
    return {
        "frame_number": frame['frame_number'],
        "full_text": ' '.join([ann['text'] for ann in cleaned_annotations]),
        "cleaned_annotations": cleaned_annotations
    }

def process_single_frame(frame: Dict) -> Dict:
    return clean_ocr_data(frame)

def clean_and_consolidate_ocr_data(ocr_results: List[Dict]) -> List[Dict]:
    def preprocess_ocr_text(text: str) -> str:
        # Common OCR error corrections
        corrections = {
            'rn': 'm',
            'li': 'h',
            'ii': 'n',
            'ln': 'in',
        }
        
        cleaned_text = text.lower()
        for error, correction in corrections.items():
            cleaned_text = cleaned_text.replace(error, correction)
        
        return cleaned_text

    def clean_annotation(annotation: Dict) -> Dict:
        original_text = annotation['text']
        text = preprocess_ocr_text(original_text)
        
        # Remove non-alphanumeric characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Skip if the cleaned text is empty or too short
        if len(cleaned_text) <= 2:
            return None
        
        words = cleaned_text.split()
        corrected_words = []
        for word in words:
            if d.check(word):
                corrected_words.append(word)
            else:
                suggestions = d.suggest(word)
                if suggestions and fuzz.ratio(word, suggestions[0]) > 80:
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)  # Keep original if no good suggestion
        
        final_text = ' '.join(corrected_words)
        
        # Brand matching
        brand_match, brand_score = fuzzy_match_brand(final_text, min_score=85)
        
        return {
            "text": brand_match if brand_match else final_text,
            "original_text": original_text,
            "cleaned_text": final_text,
            "brand_match": brand_match,
            "brand_score": brand_score,
            "bounding_box": annotation['bounding_box']
        }

    cleaned_results = []
    for frame in ocr_results:
        cleaned_annotations = [clean_annotation(ann) for ann in frame.get('text_annotations', [])]
        cleaned_annotations = [ann for ann in cleaned_annotations if ann is not None]
        
        cleaned_frame = {
            "frame_number": frame['frame_number'],
            "full_text": ' '.join([ann['text'] for ann in cleaned_annotations]),
            "original_full_text": frame.get('full_text', ''),
            "cleaned_annotations": cleaned_annotations
        }
        cleaned_results.append(cleaned_frame)
    
    return cleaned_results

def custom_ratio(s1, s2):
    """Custom ratio function that gives more weight to the beginning of the string."""
    base_ratio = fuzz.ratio(s1, s2)
    if len(s1) >= 3 and len(s2) >= 3:
        start_ratio = fuzz.ratio(s1[:3], s2[:3])
        return (base_ratio + start_ratio) / 2
    return base_ratio

import traceback

def fuzzy_match_brand(text: str, min_score: int = 85) -> Tuple[Optional[str], int]:
    try:
        if text is None:
            logger.warning("fuzzy_match_brand received None as input text")
            return None, 0
        
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower()).strip()
        if len(cleaned_text) < 3:
            logger.debug(f"Text too short for brand matching: '{text}'")
            return None, 0
        
        logger.debug(f"Attempting to match brand for text: '{text}', cleaned: '{cleaned_text}'")
        
        # Check for exact match first
        if cleaned_text in BRAND_DATABASE:
            logger.debug(f"Exact match found for '{cleaned_text}'")
            return cleaned_text, 100
        
        # Check for partial matches in variations
        for brand, data in BRAND_DATABASE.items():
            variations = data.get('variations', [])
            if not isinstance(variations, list):
                logger.warning(f"Unexpected 'variations' format for brand '{brand}': {variations}")
                continue
            
            if cleaned_text in variations:
                logger.debug(f"Variation match found for '{cleaned_text}' in brand '{brand}'")
                return brand, 95
            
            for variation in variations:
                if cleaned_text in variation or variation in cleaned_text:
                    logger.debug(f"Partial variation match found for '{cleaned_text}' in brand '{brand}'")
                    return brand, 90
        
        # Use custom ratio for fuzzy matching
        all_brand_variations = []
        for brand, data in BRAND_DATABASE.items():
            variations = data.get('variations', [])
            if isinstance(variations, list):
                all_brand_variations.extend([(brand, variation) for variation in [brand] + variations])
            else:
                logger.warning(f"Skipping invalid variations for brand '{brand}': {variations}")
        
        if not all_brand_variations:
            logger.warning("No valid brand variations found for fuzzy matching")
            return None, 0
        
        matching_variations = [variation for _, variation in all_brand_variations if abs(len(variation) - len(cleaned_text)) <= 2]
        
        if not matching_variations:
            logger.debug(f"No length-appropriate variations found for '{cleaned_text}'")
            return None, 0
        
        best_match, score = process.extractOne(cleaned_text, matching_variations, scorer=custom_ratio)
        
        if score >= min_score:
            matched_brand = next((brand for brand, variation in all_brand_variations if variation == best_match), None)
            if matched_brand:
                logger.debug(f"Fuzzy match for '{text}': Best match '{best_match}' (brand: {matched_brand}) with score {score}")
                return matched_brand, score
            else:
                logger.warning(f"Matched variation '{best_match}' not found in brand database")
        else:
            logger.debug(f"No fuzzy match found for '{text}' above minimum score {min_score}")
        
        return None, 0
    except Exception as e:
        logger.error(f"Error in fuzzy_match_brand for text '{text}': {str(e)}")
        logger.error(traceback.format_exc())
        return None, 0

def are_words_close(word1: Dict, word2: Dict, horizontal_threshold: int = 100, vertical_threshold: int = 50) -> bool:
    """
    Check if two words are close enough to be considered part of the same brand name.
    Proximity is measured as the distance between the bounding boxes.
    """
    box1 = word1['bounding_box']
    box2 = word2['bounding_box']

    # Calculate the horizontal and vertical distances between the two bounding boxes
    hor_distance = min(abs(box1['vertices'][1]['x'] - box2['vertices'][0]['x']),
                       abs(box2['vertices'][1]['x'] - box1['vertices'][0]['x']))
    vert_distance = min(abs(box1['vertices'][0]['y'] - box2['vertices'][0]['y']),
                        abs(box1['vertices'][2]['y'] - box2['vertices'][2]['y']))

    is_close = hor_distance < horizontal_threshold and vert_distance < vertical_threshold
    logger.debug(f"Words '{word1['text']}' and '{word2['text']}' - Horizontal distance: {hor_distance}, Vertical distance: {vert_distance}, Is close: {is_close}")
    return is_close

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
    consolidated = []
    i = 0
    while i < len(words):
        current_word = words[i]
        # Check if the current word alone matches a brand
        match, score = fuzzy_match_brand(current_word['text'])
        if match:
            current_word['text'] = match
            current_word['confidence'] = score
            consolidated.append(current_word)
            i += 1
            continue

        # If not, try to combine with the next word
        if i + 1 < len(words):
            next_word = words[i + 1]
            if are_words_close(current_word, next_word):
                combined_text = f"{current_word['text']} {next_word['text']}"
                match, score = fuzzy_match_brand(combined_text)
                if match:
                    combined_word = {
                        "text": match,
                        "original_text": combined_text,
                        "confidence": score,
                        "bounding_box": merge_bounding_boxes(current_word['bounding_box'], next_word['bounding_box'])
                    }
                    # logger.info(f"Consolidated words: '{current_word['text']}' and '{next_word['text']}' into '{match}' with confidence {score}")
                    consolidated.append(combined_word)
                    i += 2
                    continue

        # If no match found, add the current word as is
        consolidated.append(current_word)
        i += 1

    return consolidated

def interpolate_brand(current_brand: Dict, next_brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    t = (frame_number - current_frame) / (next_frame - current_frame)
    
    # Check if bounding boxes exist before interpolating
    if 'bounding_box' in current_brand and 'bounding_box' in next_brand:
        interpolated_box = {
            'vertices': [
                {k: int(current_brand['bounding_box']['vertices'][j][k] + 
                         t * (next_brand['bounding_box']['vertices'][j][k] - 
                              current_brand['bounding_box']['vertices'][j][k]))
                 for k in ['x', 'y']}
                for j in range(4)
            ]
        }
    else:
        interpolated_box = None

    confidence_decay = np.exp(-0.5 * (frame_number - current_frame) / fps)
    interpolated_confidence = current_brand['confidence'] * confidence_decay

    return {
        "text": current_brand['text'],
        "original_text": current_brand['text'],
        "confidence": interpolated_confidence,
        "bounding_box": interpolated_box,
        "is_interpolated": True
    }

def detect_brands_and_interpolate(cleaned_results: List[Dict], fps: float, video_resolution: Tuple[int, int]) -> Tuple[List[Dict], Dict[str, Set[int]]]:
    video_width, video_height = video_resolution

    # Configurable parameters
    params = {
        "high_confidence_threshold": 90,    # Minimum score for high-confidence detections
        "low_confidence_threshold": 80,     # Minimum score for low-confidence detections
        "min_brand_length": 3,              # Minimum length of a brand name
        "min_detections": 2,                # Minimum number of detections for a brand to be considered
        "frame_window": int(fps * 1),       # Window (in frames) for checking brand consistency
        "text_difference_threshold": 60,    # Minimum fuzzy match score between original and matched text
        "min_text_width_pct": 5,            # Minimum text width as percentage of video width
        "min_text_height_pct": 5,           # Minimum text height as percentage of video height
        "interpolation_confidence": 70,     # Confidence score for interpolated brand appearances
    }

    # Calculate minimum text dimensions in pixels
    min_text_width = int(video_width * params["min_text_width_pct"] / 100)
    min_text_height = int(video_height * params["min_text_height_pct"] / 100)

    brand_results = []
    brand_appearances = defaultdict(list)

    # First pass: Detect brands
    for frame in cleaned_results:
        frame_number = frame['frame_number']
        detected_brands = []
        for annotation in frame['cleaned_annotations']:
            if (annotation['brand_match'] and 
                annotation['brand_score'] >= params["low_confidence_threshold"] and
                len(annotation['brand_match']) >= params["min_brand_length"]):
                
                # Check for difference between original and matched text
                if fuzz.ratio(annotation['original_text'].lower(), annotation['brand_match'].lower()) < params["text_difference_threshold"]:
                    continue

                # Check text size
                box = annotation['bounding_box']
                width = max(v['x'] for v in box['vertices']) - min(v['x'] for v in box['vertices'])
                height = max(v['y'] for v in box['vertices']) - min(v['y'] for v in box['vertices'])
                if width < min_text_width or height < min_text_height:
                    continue

                detected_brands.append({
                    'text': annotation['brand_match'],
                    'confidence': annotation['brand_score'],
                    'bounding_box': annotation['bounding_box'],
                    'original_text': annotation['original_text'],
                    'cleaned_text': annotation['cleaned_text']
                })
                brand_appearances[annotation['brand_match']].append((frame_number, annotation['brand_score']))

        brand_results.append({
            "frame_number": frame_number,
            "detected_brands": detected_brands
        })

    # Filter out brands with too few detections
    brand_appearances = {brand: appearances for brand, appearances in brand_appearances.items() 
                         if len(appearances) >= params["min_detections"]}

    # Second pass: Interpolate brands
    final_brand_appearances = {}
    for brand, appearances in brand_appearances.items():
        appearances.sort(key=lambda x: x[0])  # Sort by frame number
        
        # Check for consistency within the frame window
        consistent_appearances = []
        for i, (frame, conf) in enumerate(appearances):
            window = [a for a in appearances if abs(a[0] - frame) <= params["frame_window"]]
            if len(window) >= params["min_detections"] or any(c >= params["high_confidence_threshold"] for _, c in window):
                consistent_appearances.append((frame, conf))

        if consistent_appearances:
            first_appearance = consistent_appearances[0][0]
            last_appearance = consistent_appearances[-1][0]
            
            # Include all frames between first and last appearance
            final_brand_appearances[brand] = set(range(first_appearance, last_appearance + 1))

            # Update brand_results
            for frame_number in range(first_appearance, last_appearance + 1):
                frame_index = frame_number - cleaned_results[0]['frame_number']
                if frame_index < len(brand_results):
                    brand_data = next((b for b in brand_results[frame_index]['detected_brands'] if b['text'] == brand), None)
                    if not brand_data:
                        # Add interpolated data
                        brand_results[frame_index]['detected_brands'].append({
                            'text': brand,
                            'confidence': params["interpolation_confidence"],
                            'is_interpolated': True
                        })

    # Sort detected brands in each frame by confidence
    for frame in brand_results:
        frame['detected_brands'].sort(key=lambda x: x['confidence'], reverse=True)

    return brand_results, final_brand_appearances

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

def create_word_cloud(s3_client, video_id: str, cleaned_results: List[Dict]):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font.
    """
    # Extract all text annotations
    all_text = []
    for frame in cleaned_results:
        for annotation in frame['cleaned_annotations']:
            # Use the cleaned_text instead of text to avoid brand names dominating the cloud
            all_text.append(annotation['cleaned_text'].lower())

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

    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/wordcloud.jpg',
            Body=img_buffer.getvalue(),
            ContentType='image/jpeg'
        )
        logger.info(f"Word cloud created and saved for video: {video_id}")
    except ClientError as e:
        logger.error(f"Error saving word cloud for video {video_id}: {str(e)}")
        raise

    plt.close()

async def process_single_frame_ocr(s3_client, video_id, frame_number):
    try:
        # Get the frame from S3
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=f'{video_id}/frames/{frame_number:06d}.jpg')
        image_content = response['Body'].read()

        # Create a client (assuming you're still using Google Vision API for OCR)
        client = vision.ImageAnnotatorClient()

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
    
def filter_brand_results(brand_results: List[Dict], brand_appearances: Dict[str, Set[int]], fps: float) -> List[Dict]:
    min_frames = int(fps)  # Minimum number of frames (1 second)
    valid_brands = {brand for brand, appearances in brand_appearances.items() if len(appearances) >= min_frames}
    
    filtered_results = []
    for frame in brand_results:
        filtered_brands = [brand for brand in frame['detected_brands'] if brand['text'] in valid_brands]
        filtered_results.append({
            "frame_number": frame['frame_number'],
            "detected_brands": filtered_brands
        })
    
    return filtered_results

async def post_process_ocr(video_id: str, fps: float, video_resolution: Tuple[int, int], s3_client):
    try:
        # Load OCR results
        ocr_results = load_ocr_results(s3_client, video_id)
        logger.info(f"Loaded {len(ocr_results)} OCR results for video: {video_id}")

        # Step 1: Clean and consolidate OCR data
        logger.info(f"Cleaning and consolidating OCR data for video: {video_id}")
        cleaned_results = clean_and_consolidate_ocr_data(ocr_results)
        logger.info(f"Cleaned and consolidated {len(cleaned_results)} frames for video: {video_id}")
        await save_processed_ocr_results(s3_client, video_id, cleaned_results)

        # Step 2: Create word cloud
        logger.info(f"Creating word cloud for video: {video_id}")
        create_word_cloud(s3_client, video_id, cleaned_results)

        # Step 3: Detect brands and interpolate
        logger.info(f"Detecting brands and interpolating for video: {video_id}")
        brand_results, brand_appearances = detect_brands_and_interpolate(cleaned_results, fps, video_resolution)
        logger.info(f"Detected {len(brand_appearances)} unique brands for video: {video_id}")

        # Step 4: Filter brand results
        logger.info(f"Filtering brand results for video: {video_id}")
        filtered_brand_results = filter_brand_results(brand_results, brand_appearances, fps)

        # Count unique brands in filtered results
        unique_brands = set()
        for frame in filtered_brand_results:
            unique_brands.update(brand['text'] for brand in frame['detected_brands'])
        logger.info(f"Filtered to {len(unique_brands)} unique brands for video: {video_id}")

        # Step 5: Save filtered brands OCR results
        await save_brands_ocr_results(s3_client, video_id, filtered_brand_results)

        # Step 6: Create and save brand table
        brand_stats = create_and_save_brand_table(s3_client, video_id, brand_appearances, fps)
        logger.info(f"Created brand table with {len(brand_stats)} entries for video: {video_id}")

        logger.info(f"Completed post-processing OCR for video: {video_id}")
        return brand_stats
    except Exception as e:
        logger.error(f"Error in post_process_ocr for video {video_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def process_ocr(video_id: str, s3_client, status_tracker: 'StatusTracker'):
    logger.info(f"Starting OCR processing for video: {video_id}")
    status_tracker.update_process_status("ocr", "in_progress", 0)

    ocr_start_time = time.time()
    
    # List frames in S3
    frame_objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=PROCESSING_BUCKET, Prefix=f'{video_id}/frames/'):
        frame_objects.extend(page.get('Contents', []))
    
    total_frames = len(frame_objects)

    ocr_results = []
    processed_frames = 0
    total_words = 0

    # Process frames in batches to limit concurrency
    batch_size = 10
    for i in range(0, len(frame_objects), batch_size):
        batch = frame_objects[i:i+batch_size]
        tasks = [process_single_frame_ocr(s3_client, video_id, int(frame['Key'].split('/')[-1].split('.')[0])) for frame in batch]
        batch_results = await asyncio.gather(*tasks)

        for result in batch_results:
            if result:
                ocr_results.append(result)
                total_words += len(result['full_text'].split())

        processed_frames += len(batch)
        progress = (processed_frames / total_frames) * 100
        status_tracker.update_process_status("ocr", "in_progress", progress)

        # Log progress periodically
        if processed_frames % 100 == 0 or processed_frames == total_frames:
            logger.info(f"Processed {processed_frames}/{total_frames} frames for video {video_id}")

    # Sort OCR results by frame number
    ocr_results.sort(key=lambda x: x['frame_number'])

    # Store OCR results
    try:
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/ocr_results.json',
            Body=json.dumps(ocr_results, indent=2),
            ContentType='application/json'
        )
    except ClientError as e:
        logger.error(f"Error saving OCR results for video {video_id}: {str(e)}")
        raise

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