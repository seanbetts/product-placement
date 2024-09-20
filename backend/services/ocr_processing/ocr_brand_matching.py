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
            # logger.debug(f"Text too short for brand matching: '{text}'")
            return None, 0
        
        # logger.debug(f"Attempting to match brand for text: '{text}', cleaned: '{cleaned_text}'")
        
        # Check for exact match first
        if cleaned_text in settings.BRAND_DATABASE:
            # logger.debug(f"Exact match found for '{cleaned_text}'")
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
def detect_brands_and_interpolate(cleaned_results: List[Dict], fps: float, video_resolution: Tuple[int, int]) -> Tuple[List[Dict], Dict[str, Dict[int, List[Dict]]]]:
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
        "interpolation_limit": settings.INTERPOLATION_LIMIT,
    }

    # Calculate minimum text dimensions in pixels
    min_text_width = video_width * params["min_text_width_pct"] / 100
    min_text_height = video_height * params["min_text_height_pct"] / 100

    brand_results = []
    brand_appearances = defaultdict(lambda: defaultdict(list))

    # logger.info("Starting first pass: Detect brands")
    # First pass: Detect brands
    for frame in cleaned_results:
        frame_number = frame['frame_number']
        # logger.info(f"Processing frame {frame_number}")
        detected_brands = []

        for detection in frame['cleaned_detections']:
            # logger.debug(f"Examining detection: {detection['text']}")

            if not detection.get('brand_match'):
                # logger.debug(f"No brand match for: {detection['text']}")
                continue

            if detection.get('brand_score', 0) < params["low_confidence_threshold"]:
                # logger.debug(f"Brand score {detection.get('brand_score')} below threshold {params['low_confidence_threshold']}")
                continue

            text_diff_score = fuzz.ratio(detection.get('original_text', '').lower(), detection['brand_match'].lower())
            if text_diff_score < params["text_difference_threshold"]:
                # logger.debug(f"Text difference score {text_diff_score} below threshold {params['text_difference_threshold']}")
                continue

            box = detection.get('bounding_box', {})
            if 'vertices' in box:
                width = max(v['x'] for v in box['vertices']) - min(v['x'] for v in box['vertices'])
                height = max(v['y'] for v in box['vertices']) - min(v['y'] for v in box['vertices'])
                width_px = width * video_width
                height_px = height * video_height
                if width_px < min_text_width or height_px < min_text_height:
                    # logger.debug(f"Text size ({width_px}x{height_px}) below minimum threshold ({min_text_width}x{min_text_height})")
                    continue
            else:
                # logger.debug("No bounding box found")
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
        # logger.info(f"Frame {frame_number}: detected {len(detected_brands)} brands")

    # logger.info("Starting second pass: Interpolate brands")
    # Second pass: Interpolate brands
    final_brand_appearances = defaultdict(lambda: defaultdict(list))
    for brand, frame_appearances in brand_appearances.items():
        # logger.info(f"Processing brand: {brand}")
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
            # logger.info(f"Brand {brand} appears from frame {first_appearance} to {last_appearance}")
            
            for frame_number in range(first_appearance, last_appearance + 1):
                if frame_number in frame_appearances:
                    final_brand_appearances[brand][frame_number] = frame_appearances[frame_number]
                    # logger.debug(f"Frame {frame_number}: {len(frame_appearances[frame_number])} instances of {brand}")
                else:
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
                        
                        frame_index = frame_number - cleaned_results[0]['frame_number']
                        if frame_index < len(brand_results):
                            brand_results[frame_index]['detected_brands'].extend(interpolated_instances)
                            final_brand_appearances[brand][frame_number] = interpolated_instances
                            # logger.debug(f"Frame {frame_number}: Interpolated {len(interpolated_instances)} instances of {brand}")

    # logger.info("Starting post-processing")
    # Post-processing to remove over-interpolated sections
    brand_results, final_brand_appearances = remove_over_interpolated_sections(brand_results, final_brand_appearances)

    # logger.info("Updating brand_results")
    # Update brand_results to include all instances from final_brand_appearances
    for frame in brand_results:
        frame_number = frame['frame_number']
        frame['detected_brands'] = []
        for brand, appearances in final_brand_appearances.items():
            if frame_number in appearances:
                frame['detected_brands'].extend(appearances[frame_number])
        # logger.debug(f"Frame {frame_number}: Final count of {len(frame['detected_brands'])} brands")

    # Sort detected brands in each frame by confidence, preserving multiple detections
    for frame in brand_results:
        frame['detected_brands'].sort(key=lambda x: x['confidence'], reverse=True)

    logger.info(f"Final result: {len(final_brand_appearances)} brands detected and interpolated")

    return brand_results, final_brand_appearances

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
def match_brand_instances(prev_instances: List[Dict], next_instances: List[Dict]) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    matched_pairs = []
    unmatched_prev = prev_instances.copy()
    unmatched_next = next_instances.copy()

    # Match instances based on position and confidence
    for prev in prev_instances:
        if unmatched_next:
            closest = min(unmatched_next, 
                          key=lambda next: (calculate_distance(prev['bounding_box'], next['bounding_box']),
                                            abs(prev['confidence'] - next['confidence'])))
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
            "cleaned_text": brand.get('cleaned_text', brand['text']),
            "confidence": brand['confidence'] * confidence_decay,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }
    except Exception as e:
        logger.error(f"Error in maintain_brand: {str(e)}")
        return None
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
            "cleaned_text": brand.get('cleaned_text', brand['text']),
            "confidence": brand['confidence'] * confidence_increase,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }
    except Exception as e:
        logger.error(f"Error in fade_in_brand: {str(e)}")
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
        "cleaned_text": brand.get('cleaned_text', brand['text']),
        "confidence": brand['confidence'] * confidence_decay,
        "bounding_box": brand['bounding_box'],
        "is_interpolated": True,
        "frame_number": frame_number
    }
########################################################

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
        "cleaned_text": prev.get('cleaned_text', prev['text']),  # Add this line
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

## Removes sections where a brand has been over interpolated
########################################################
def remove_over_interpolated_sections(brand_results: List[Dict], final_brand_appearances: Dict[str, Dict[int, List[Dict]]]) -> Tuple[List[Dict], Dict[str, Dict[int, List[Dict]]]]:
    # Get unique brand names
    brand_names = set(brand['text'] for frame in brand_results for brand in frame['detected_brands'])
    
    for brand_name in brand_names:
        interpolated_sequences = find_interpolated_sequences(brand_results, brand_name)
        
        for start, end in interpolated_sequences:
            if end - start + 1 > settings.INTERPOLATION_LIMIT:
                # Remove all interpolated brands in this sequence
                for frame in brand_results[start-brand_results[0]['frame_number']:end-brand_results[0]['frame_number']+1]:
                    frame['detected_brands'] = [b for b in frame['detected_brands'] if b['text'] != brand_name or not b.get('is_interpolated', False)]
                
                # Update final_brand_appearances
                for frame_num in range(start, end + 1):
                    if frame_num in final_brand_appearances[brand_name]:
                        final_brand_appearances[brand_name][frame_num] = [
                            b for b in final_brand_appearances[brand_name][frame_num]
                            if not b.get('is_interpolated', False)
                        ]
                        if not final_brand_appearances[brand_name][frame_num]:
                            del final_brand_appearances[brand_name][frame_num]

    # Remove empty brand entries
    final_brand_appearances = {k: v for k, v in final_brand_appearances.items() if v}

    return brand_results, final_brand_appearances
########################################################

## Find sequences where a brand has been interpolated too many times
########################################################
def find_interpolated_sequences(brand_results: List[Dict], brand_name: str) -> List[Tuple[int, int]]:
    sequences = []
    start = None
    for frame in brand_results:
        interpolated_brands = [b for b in frame['detected_brands'] if b['text'] == brand_name and b.get('is_interpolated', False)]
        if interpolated_brands:
            if start is None:
                start = frame['frame_number']
        elif start is not None:
            sequences.append((start, frame['frame_number'] - 1))
            start = None
    
    if start is not None:
        sequences.append((start, brand_results[-1]['frame_number']))
    
    return sequences
########################################################
## Detects if a brand has been interpolated too many times consecutively
########################################################
def is_over_interpolated(brand: Dict, brand_results: List[Dict], current_frame: int) -> bool:
    if not brand.get('is_interpolated', False):
        return False

    brand_name = brand['text']
    sequences = find_interpolated_sequences(brand_results, brand_name)
    
    for start, end in sequences:
        if start <= current_frame <= end and end - start + 1 > settings.INTERPOLATION_LIMIT:
            return True
    
    return False
########################################################