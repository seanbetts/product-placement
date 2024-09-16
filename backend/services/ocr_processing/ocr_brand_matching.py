import re
import traceback
from typing import Tuple, Optional, List, Dict, Set
from collections import defaultdict
from thefuzz import fuzz, process
from core.logging import logger
from core.config import settings
from utils import utils

## Fuzzy match brands
########################################################
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
        if cleaned_text in settings.BRAND_DATABASE:
            logger.debug(f"Exact match found for '{cleaned_text}'")
            return cleaned_text, 100
        
        # Check for partial matches in variations
        for brand, data in settings.BRAND_DATABASE.items():
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
        for brand, data in settings.BRAND_DATABASE.items():
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
        
        best_match, score = process.extractOne(cleaned_text, matching_variations, scorer=utils.custom_ratio)
        
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
########################################################

## Find brands and interpolate appearances
########################################################
def detect_brands_and_interpolate(cleaned_results: List[Dict], fps: float, video_resolution: Tuple[int, int]) -> Tuple[List[Dict], Dict[str, Set[int]]]:
    video_width, video_height = video_resolution

    # Configurable parameters
    params = {
        "high_confidence_threshold": settings.HIGH_CONFIDENCE_THRESHOLD,
        "low_confidence_threshold": settings.LOW_CONFIDENCE_THRESHOLD,
        "min_brand_length": settings.MIN_BRAND_LENGTH,
        "min_detections": settings.MIN_DETECTIONS,
        "frame_window": int(fps * settings.FRAME_WINDOW),
        "text_difference_threshold": settings.TEXT_DIFF_THRESHOLD,
        "min_text_width_pct": settings.MIN_TEXT_WIDTH,
        "min_text_height_pct": settings.MIN_TEXT_HEIGHT,
        "interpolation_confidence": settings.INTERPOLATION_CONFIDENCE, 
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
########################################################