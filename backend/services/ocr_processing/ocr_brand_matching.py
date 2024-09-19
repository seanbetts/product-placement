import re
import traceback
import numpy as np
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
        
        # logger.debug(f"Attempting to match brand for text: '{text}', cleaned: '{cleaned_text}'")
        
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
                # logger.debug(f"Variation match found for '{cleaned_text}' in brand '{brand}'")
                return brand, 95
            
            for variation in variations:
                if cleaned_text in variation or variation in cleaned_text:
                    # logger.debug(f"Partial variation match found for '{cleaned_text}' in brand '{brand}'")
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
            # logger.debug(f"No length-appropriate variations found for '{cleaned_text}'")
            return None, 0
        
        best_match, score = process.extractOne(cleaned_text, matching_variations, scorer=utils.custom_ratio)
        
        if score >= min_score:
            matched_brand = next((brand for brand, variation in all_brand_variations if variation == best_match), None)
            if matched_brand:
                # logger.debug(f"Fuzzy match for '{text}': Best match '{best_match}' (brand: {matched_brand}) with score {score}")
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
    min_text_width = video_width * params["min_text_width_pct"] / 100
    min_text_height = video_height * params["min_text_height_pct"] / 100

    # logger.info(f"Minimum text dimensions: width={min_text_width:.2f}px, height={min_text_height:.2f}px")

    brand_results = []
    brand_appearances = defaultdict(lambda: defaultdict(list))

    # First pass: Detect brands
    total_detections = 0
    for frame in cleaned_results:
        frame_number = frame['frame_number']
        detected_brands = []
        seen_texts = set()  # To avoid duplicates

        # logger.info(f"Processing frame {frame_number}")
        # logger.info(f"Number of cleaned detections: {len(frame['cleaned_detections'])}")

        for detection in frame['cleaned_detections']:
            total_detections += 1
            # Skip if we've already processed this text (to avoid LINE/WORD duplication)
            if detection['text'] in seen_texts:
                # logger.debug(f"Skipping duplicate text: {detection['text']}")
                continue
            seen_texts.add(detection['text'])

            # logger.debug(f"Examining detection: {detection['text']}")
            # logger.debug(f"Brand match: {detection.get('brand_match')}, Brand score: {detection.get('brand_score')}")

            if not detection.get('brand_match'):
                # logger.debug("No brand match found, skipping")
                continue

            if detection.get('brand_score', 0) < params["low_confidence_threshold"]:
                # logger.debug(f"Brand score {detection.get('brand_score')} below threshold {params['low_confidence_threshold']}, skipping")
                continue

            # Check for difference between original and matched text
            text_diff_score = fuzz.ratio(detection.get('original_text', '').lower(), detection['brand_match'].lower())
            # logger.debug(f"Text difference score: {text_diff_score}")
            if text_diff_score < params["text_difference_threshold"]:
                # logger.debug(f"Text difference score below threshold {params['text_difference_threshold']}, skipping")
                continue

            # Check text size
            box = detection.get('bounding_box', {})
            if 'vertices' in box:
                width = max(v['x'] for v in box['vertices']) - min(v['x'] for v in box['vertices'])
                height = max(v['y'] for v in box['vertices']) - min(v['y'] for v in box['vertices'])
                width_px = width * video_width
                height_px = height * video_height
                # logger.debug(f"Text dimensions: width={width_px:.2f}px, height={height_px:.2f}px")
                if width_px < min_text_width or height_px < min_text_height:
                    logger.debug("Text size below minimum threshold, skipping")
                    continue
            else:
                # logger.debug("No bounding box found, skipping")
                continue

            # logger.info(f"Brand detected: {detection['brand_match']}")
            brand_data = {
                'text': detection['brand_match'],
                'confidence': detection['brand_score'],
                'bounding_box': box,
                'original_text': detection.get('original_text', ''),
                'cleaned_text': detection.get('cleaned_text', ''),
                'frame_number': frame_number
            }
            detected_brands.append(brand_data)
            brand_appearances[detection['brand_match']][frame_number].append(brand_data)

        brand_results.append({
            "frame_number": frame_number,
            "detected_brands": detected_brands
        })

    # logger.info(f"Total detections processed: {total_detections}")
    # logger.info(f"Initial brand detection: {len(brand_appearances)} potential brands found")
    # for brand, appearances in brand_appearances.items():
    #     logger.info(f"Brand '{brand}' detected in {len(appearances)} frames")

    # Second pass: Interpolate brands
    final_brand_appearances = {}
    for brand, frame_appearances in brand_appearances.items():
        all_frames = sorted(frame_appearances.keys())
        consistent_appearances = []
        for frame in all_frames:
            window = [f for f in all_frames if abs(f - frame) <= params["frame_window"]]
            if (len(window) >= params["min_detections"] or 
                any(any(instance['confidence'] >= params["high_confidence_threshold"] for instance in frame_appearances[f]) for f in window)):
                consistent_appearances.extend(window)
        
        consistent_appearances = sorted(set(consistent_appearances))
        
        if consistent_appearances:
            first_appearance = consistent_appearances[0]
            last_appearance = consistent_appearances[-1]
            
            final_brand_appearances[brand] = set(range(first_appearance, last_appearance + 1))

            # Interpolate missing frames
            for frame_number in range(first_appearance, last_appearance + 1):
                if frame_number not in frame_appearances:
                    # Find nearest previous and next frames with detections
                    prev_frame = max((f for f in all_frames if f < frame_number), default=None)
                    next_frame = min((f for f in all_frames if f > frame_number), default=None)
                    
                    if prev_frame is not None and next_frame is not None:
                        prev_instances = frame_appearances[prev_frame]
                        next_instances = frame_appearances[next_frame]
                        
                        interpolated_instances = interpolate_brand_instances(
                            prev_instances, next_instances,
                            prev_frame, next_frame,
                            frame_number, fps
                        )
                        
                        # Add interpolated instances to brand_results
                        frame_index = frame_number - cleaned_results[0]['frame_number']
                        if frame_index < len(brand_results):
                            brand_results[frame_index]['detected_brands'].extend(interpolated_instances)

    # Sort detected brands in each frame by confidence
    for frame in brand_results:
        frame['detected_brands'].sort(key=lambda x: x['confidence'], reverse=True)

    logger.info(f"Final result: {len(final_brand_appearances)} brands detected and interpolated")

    return brand_results, final_brand_appearances

## Calculate bounding boxes for interpolated brands
########################################################
def interpolate_brand(prev: Dict, next: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    t = (frame_number - current_frame) / (next_frame - current_frame)
    
    # Interpolate bounding box
    interpolated_box = {
        'vertices': [
            {k: int(prev['bounding_box']['vertices'][i][k] + 
                    t * (next['bounding_box']['vertices'][i][k] - prev['bounding_box']['vertices'][i][k]))
             for k in ['x', 'y']}
            for i in range(4)
        ]
    }
    
    # Interpolate confidence
    confidence = prev['confidence'] + t * (next['confidence'] - prev['confidence'])
    
    return {
        "text": prev['text'],
        "original_text": prev.get('original_text', prev['text']),
        "confidence": confidence,
        "bounding_box": interpolated_box,
        "is_interpolated": True,
        "frame_number": frame_number
    }
########################################################

## Calculate distances between two bounding boxes
########################################################
def calculate_distance(box1: Dict, box2: Dict) -> float:
    """Calculate the Euclidean distance between the centers of two bounding boxes."""
    center1 = np.mean([[v['x'], v['y']] for v in box1['vertices']], axis=0)
    center2 = np.mean([[v['x'], v['y']] for v in box2['vertices']], axis=0)
    return np.linalg.norm(center1 - center2)
########################################################

## Interpolate instances where brands appear 2+ times in a single frame
########################################################
def interpolate_brand_instances(prev_instances: List[Dict], next_instances: List[Dict],
                                current_frame: int, next_frame: int, frame_number: int, fps: float) -> List[Dict]:
    try:
        matched_pairs = match_brand_instances(prev_instances, next_instances)
        interpolated_instances = []
        
        for prev, next in matched_pairs:
            try:
                if prev and next:
                    # Both instances exist, interpolate normally
                    interpolated = interpolate_brand(prev, next, current_frame, next_frame, frame_number, fps)
                elif prev:
                    # Only previous instance exists, maintain with slight decay
                    interpolated = maintain_brand(prev, current_frame, next_frame, frame_number, fps)
                elif next:
                    # Only next instance exists, fade in
                    interpolated = fade_in_brand(next, current_frame, next_frame, frame_number, fps)
                
                if interpolated:
                    interpolated_instances.append(interpolated)
            except Exception as e:
                logger.warning(f"Error interpolating brand instance: {str(e)}")
                continue
        
        return interpolated_instances
    except Exception as e:
        logger.error(f"Error in interpolate_brand_instances: {str(e)}")
        return []
########################################################

## Detect when brands appear multiple times in a single frame
########################################################
def match_brand_instances(prev_instances: List[Dict], next_instances: List[Dict]) -> List[Tuple[Dict, Dict]]:
    matched_pairs = []
    unmatched_prev = prev_instances.copy()
    unmatched_next = next_instances.copy()

    # Match instances based on position
    for prev in prev_instances:
        if unmatched_next:
            closest = min(unmatched_next, key=lambda next: calculate_distance(prev['bounding_box'], next['bounding_box']))
            matched_pairs.append((prev, closest))
            unmatched_prev.remove(prev)
            unmatched_next.remove(closest)
        else:
            matched_pairs.append((prev, None))

    # Add remaining unmatched next instances
    for next in unmatched_next:
        matched_pairs.append((None, next))

    return matched_pairs
########################################################

## Maintain brands across frames
########################################################
def maintain_brand(brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    try:
        t = (frame_number - current_frame) / (next_frame - current_frame)
        confidence_decay = np.exp(-0.5 * t)  # Slower decay
        return {
            "text": brand['text'],
            "original_text": brand.get('original_text', brand['text']),
            "cleaned_text": brand.get('cleaned_text', brand['text']),  # Add this line
            "confidence": brand['confidence'] * confidence_decay,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }
    except Exception as e:
        logger.error(f"Error in maintain_brand: {str(e)}")
        return None
########################################################

## Handle brands fading out from frames
########################################################
def fade_out_brand(brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    t = (frame_number - current_frame) / (next_frame - current_frame)
    confidence_decay = np.exp(-2 * t)  # Faster decay for fading out
    
    return {
        "text": brand['text'],
        "original_text": brand.get('original_text', brand['text']),
        "confidence": brand['confidence'] * confidence_decay,
        "bounding_box": brand['bounding_box'],
        "is_interpolated": True,
        "frame_number": frame_number
    }
########################################################

## Handle brands fading in to frames
########################################################
def fade_in_brand(brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    try:
        t = (frame_number - current_frame) / (next_frame - current_frame)
        confidence_increase = 1 - np.exp(-2 * t)  # Faster increase for fading in
        return {
            "text": brand['text'],
            "original_text": brand.get('original_text', brand['text']),
            "cleaned_text": brand.get('cleaned_text', brand['text']),  # Add this line
            "confidence": brand['confidence'] * confidence_increase,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }
    except Exception as e:
        logger.error(f"Error in fade_in_brand: {str(e)}")
        return None
########################################################