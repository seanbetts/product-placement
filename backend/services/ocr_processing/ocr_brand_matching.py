import re
import asyncio
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import defaultdict
from thefuzz import fuzz, process
from core.config import settings
from core.logging import logger
from utils import utils

## Find brands and interpolate appearances
########################################################
async def detect_brands_and_interpolate(cleaned_results: List[Dict], fps: float, video_resolution: Tuple[int, int]) -> Tuple[List[Dict], Dict[str, Dict[int, List[Dict]]]]:
    try:
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

        logger.info("Video Processing - Brand Detection - Step 3.4.2: Starting first pass - Detecting brands")
        
        # First pass: Detect brands
        for frame in cleaned_results:
            frame_number = frame['frame_number']
            detected_brands = []

            for detection in frame['cleaned_detections']:
                logger.debug(f"Video Processing - Brand Detection - Step 3.4.2: Examining detection: {detection['text']}")

                if not detection.get('brand_match'):
                    logger.debug(f"Video Processing - Brand Detection - Step 3.4.2: No brand match for: {detection['text']}")
                    continue

                if detection.get('brand_score', 0) < params["low_confidence_threshold"]:
                    logger.debug(f"Video Processing - Brand Detection - Step 3.4.2: Brand score {detection.get('brand_score')} below threshold {params['low_confidence_threshold']}")
                    continue

                text_diff_score = fuzz.ratio(detection.get('original_text', '').lower(), detection['brand_match'].lower())
                if text_diff_score < params["text_difference_threshold"]:
                    logger.debug(f"Video Processing - Brand Detection - Step 3.4.2: Text difference score {text_diff_score} below threshold {params['text_difference_threshold']}")
                    continue

                box = detection.get('bounding_box', {})
                if 'vertices' in box:
                    width = max(v['x'] for v in box['vertices']) - min(v['x'] for v in box['vertices'])
                    height = max(v['y'] for v in box['vertices']) - min(v['y'] for v in box['vertices'])
                    width_px = width * video_width
                    height_px = height * video_height
                    if width_px < min_text_width or height_px < min_text_height:
                        logger.debug(f"Video Processing - Brand Detection - Step 3.4.2: Text size ({width_px}x{height_px}) below minimum threshold ({min_text_width}x{min_text_height})")
                        continue
                else:
                    logger.debug("Video Processing - Brand Detection - Step 3.4.2: No bounding box found")
                    continue

                logger.debug(f"Video Processing - Brand Detection - Step 3.4.2: Brand detected: {detection['brand_match']}")
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

        
        # Second pass: Interpolate brands
        logger.info("Video Processing - Brand Detection - Step 3.4.3: Starting second pass - Interpolating brands")
        final_brand_appearances = defaultdict(lambda: defaultdict(list))
        for brand, frame_appearances in brand_appearances.items():
            logger.debug(f"Video Processing - Brand Detection - Step 3.4.3: Processing brand: {brand}")
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
                logger.debug(f"Video Processing - Brand Detection - Step 3.4.3: Brand {brand} appears from frame {first_appearance} to {last_appearance}")
                
                for frame_number in range(first_appearance, last_appearance + 1):
                    if frame_number in frame_appearances:
                        final_brand_appearances[brand][frame_number] = frame_appearances[frame_number]
                        logger.debug(f"Video Processing - Brand Detection - Step 3.4.3: Frame {frame_number}: {len(frame_appearances[frame_number])} instances of {brand}")
                    else:
                        prev_frame = max((f for f in all_frames if f < frame_number), default=None)
                        next_frame = min((f for f in all_frames if f > frame_number), default=None)
                        
                        if prev_frame is not None and next_frame is not None:
                            prev_instances = frame_appearances[prev_frame]
                            next_instances = frame_appearances[next_frame]
                            
                            interpolated_instances = await interpolate_brand_instances(prev_instances, next_instances, prev_frame, next_frame, frame_number, fps)
                            
                            frame_index = frame_number - cleaned_results[0]['frame_number']
                            if frame_index < len(brand_results):
                                brand_results[frame_index]['detected_brands'].extend(interpolated_instances)
                                final_brand_appearances[brand][frame_number] = interpolated_instances
                                logger.debug(f"Video Processing - Brand Detection - Step 3.4.3: Frame {frame_number}: Interpolated {len(interpolated_instances)} instances of {brand}")

        logger.info("Video Processing - Brand Detection - Step 3.4.4: Removing over interpolated brands")
        # Post-processing to remove over-interpolated sections
        brand_results, final_brand_appearances = await remove_over_interpolated_sections(brand_results, final_brand_appearances)

        logger.debug("Video Processing - Brand Detection - Step 3.4.4: Updating brand_results")
        # Update brand_results to include all instances from final_brand_appearances
        for frame in brand_results:
            frame_number = frame['frame_number']
            frame['detected_brands'] = []
            for brand, appearances in final_brand_appearances.items():
                if frame_number in appearances:
                    frame['detected_brands'].extend(appearances[frame_number])
            logger.debug(f"Video Processing - Brand Detection - Step 3.4.4: Frame {frame_number}: Final count of {len(frame['detected_brands'])} brands")

        # Sort detected brands in each frame by confidence, preserving multiple detections
        for frame in brand_results:
            frame['detected_brands'] = sorted(frame['detected_brands'], key=lambda x: x['confidence'], reverse=True)

        return brand_results, final_brand_appearances

    except Exception as e:
        logger.error(f"Video Processing - Brand Detection: Error in detect_brands_and_interpolate: {str(e)}", exc_info=True)
        raise
########################################################

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
                logger.debug(f"Matched variation '{best_match}' not found in brand database")
                None
        else:
            logger.debug(f"No fuzzy match found for '{text}' above minimum score {min_score}")
            None

        return None, 0

    except Exception as e:
        logger.error(f"Error in fuzzy_match_brand for text '{text}': {str(e)}", exc_info=True)
        return None, 0
########################################################

## Interpolate instances where brands appear 2+ times in a single frame
########################################################
async def interpolate_brand_instances(prev_instances: List[Dict], next_instances: List[Dict],
                                current_frame: int, next_frame: int, frame_number: int, fps: float) -> List[Dict]:
    try:
        logger.debug(f"Interpolating brand instances for frame {frame_number}")
        matched_pairs = await match_brand_instances(prev_instances, next_instances)
        interpolated_instances = []

        for prev, next in matched_pairs:
            try:
                if prev and next:
                    # Both instances exist, interpolate normally
                    logger.debug(f"Interpolating between existing instances for frame {frame_number}")
                    interpolated = await interpolate_brand(prev, next, current_frame, next_frame, frame_number, fps)
                elif prev:
                    # Only previous instance exists, maintain with slight decay
                    logger.debug(f"Maintaining previous instance with decay for frame {frame_number}")
                    interpolated = await maintain_brand(prev, current_frame, next_frame, frame_number, fps)
                elif next:
                    # Only next instance exists, fade in
                    logger.debug(f"Fading in next instance for frame {frame_number}")
                    interpolated = await fade_in_brand(next, current_frame, next_frame, frame_number, fps)
                else:
                    logger.error(f"No valid instances to interpolate for frame {frame_number}")
                    continue

                if interpolated:
                    interpolated_instances.append(interpolated)
                else:
                    logger.debug(f"Interpolation returned None for frame {frame_number}")
                    continue

            except Exception as e:
                logger.error(f"Error interpolating brand instance for frame {frame_number}: {str(e)}", exc_info=True)
                continue

        logger.debug(f"Interpolated {len(interpolated_instances)} instances for frame {frame_number}")
        return interpolated_instances

    except Exception as e:
        logger.error(f"Error in interpolate_brand_instances for frame {frame_number}: {str(e)}", exc_info=True)
        return []
########################################################

## Detect when brands appear multiple times in a single frame
########################################################
async def match_brand_instances(prev_instances: List[Dict], next_instances: List[Dict]) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    try:
        logger.debug(f"Matching brand instances: {len(prev_instances)} previous, {len(next_instances)} next")
        matched_pairs = []
        unmatched_prev = prev_instances.copy()
        unmatched_next = next_instances.copy()

        # Match instances based on position and confidence
        for prev in prev_instances:
            if unmatched_next:
                distances = await asyncio.gather(*[calculate_distance(prev['bounding_box'], next['bounding_box']) for next in unmatched_next])
                closest = min(zip(unmatched_next, distances), key=lambda x: (x[1], abs(prev['confidence'] - x[0]['confidence'])))[0]
                matched_pairs.append((prev, closest))
                unmatched_prev.remove(prev)
                unmatched_next.remove(closest)
                logger.debug(f"Matched instance: prev={prev['text']}, next={closest['text']}")
            else:
                matched_pairs.append((prev, None))
                logger.debug(f"Unmatched previous instance: {prev['text']}")

        # Add remaining unmatched next instances
        for next in unmatched_next:
            matched_pairs.append((None, next))
            logger.debug(f"Unmatched next instance: {next['text']}")

        logger.debug(f"Matched {len(matched_pairs)} pairs of instances")
        return matched_pairs

    except Exception as e:
        logger.error(f"Error in match_brand_instances: {str(e)}", exc_info=True)
        return []
########################################################

## Maintain brands across frames
########################################################
async def maintain_brand(brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Optional[Dict]:
    try:
        logger.debug(f"Maintaining brand '{brand['text']}' for frame {frame_number}")

        t = (frame_number - current_frame) / (next_frame - current_frame)
        confidence_decay = np.exp(-0.5 * t)  # Slower decay
        
        logger.debug(f"Confidence decay factor: {confidence_decay:.4f}")

        maintained_brand = {
            "text": brand['text'],
            "original_text": brand.get('original_text', brand['text']),
            "cleaned_text": brand.get('cleaned_text', brand['text']),
            "confidence": brand['confidence'] * confidence_decay,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }

        logger.debug(f"Maintained brand '{brand['text']}' for frame {frame_number} with confidence {maintained_brand['confidence']:.4f}")
        return maintained_brand

    except Exception as e:
        logger.error(f"Error in maintain_brand for frame {frame_number}: {str(e)}", exc_info=True)
        return None
########################################################

## Handle brands fading in to frames
########################################################
async def fade_in_brand(brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Optional[Dict]:
    try:
        logger.debug(f"Fading in brand '{brand['text']}' for frame {frame_number}")

        t = (frame_number - current_frame) / (next_frame - current_frame)
        confidence_increase = 1 - np.exp(-2 * t)  # Faster increase for fading in
        
        logger.debug(f"Confidence increase factor: {confidence_increase:.4f}")

        faded_in_brand = {
            "text": brand['text'],
            "original_text": brand.get('original_text', brand['text']),
            "cleaned_text": brand.get('cleaned_text', brand['text']),
            "confidence": brand['confidence'] * confidence_increase,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }

        logger.debug(f"Faded in brand '{brand['text']}' for frame {frame_number} with confidence {faded_in_brand['confidence']:.4f}")
        return faded_in_brand

    except Exception as e:
        logger.error(f"Error in fade_in_brand for frame {frame_number}: {str(e)}", exc_info=True)
        return None
########################################################

## Handle brands fading out from frames
########################################################
async def fade_out_brand(brand: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Optional[Dict]:
    try:
        logger.debug(f"Fading out brand '{brand['text']}' for frame {frame_number}")

        t = (frame_number - current_frame) / (next_frame - current_frame)
        confidence_decay = np.exp(-2 * t)  # Faster decay for fading out
        
        logger.debug(f"Confidence decay factor: {confidence_decay:.4f}")

        faded_out_brand = {
            "text": brand['text'],
            "original_text": brand.get('original_text', brand['text']),
            "cleaned_text": brand.get('cleaned_text', brand['text']),
            "confidence": brand['confidence'] * confidence_decay,
            "bounding_box": brand['bounding_box'],
            "is_interpolated": True,
            "frame_number": frame_number
        }

        logger.debug(f"Faded out brand '{brand['text']}' for frame {frame_number} with confidence {faded_out_brand['confidence']:.4f}")
        return faded_out_brand

    except Exception as e:
        logger.error(f"Error in fade_out_brand for frame {frame_number}: {str(e)}", exc_info=True)
        return None
########################################################

## Calculate bounding boxes for interpolated brands
########################################################
async def interpolate_brand(prev: Dict, next: Dict, current_frame: int, next_frame: int, frame_number: int, fps: float) -> Dict:
    try:
        logger.debug(f"Interpolating brand '{prev['text']}' for frame {frame_number}")

        t = (frame_number - current_frame) / (next_frame - current_frame)
        logger.debug(f"Interpolation factor t: {t:.4f}")

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
        
        interpolated_brand = {
            "text": prev['text'],
            "original_text": prev.get('original_text', prev['text']),
            "cleaned_text": prev.get('cleaned_text', prev['text']),
            "confidence": confidence,
            "bounding_box": interpolated_box,
            "is_interpolated": True,
            "frame_number": frame_number
        }

        logger.debug(f"Interpolated brand '{prev['text']}' for frame {frame_number} with confidence {confidence:.4f}")
        return interpolated_brand

    except Exception as e:
        logger.error(f"Error in interpolate_brand for frame {frame_number}: {str(e)}", exc_info=True)
        # In case of error, return the previous brand data as a fallback
        return {**prev, "is_interpolated": True, "frame_number": frame_number}
########################################################

## Calculate distances between two bounding boxes
########################################################
async def calculate_distance(box1: Dict, box2: Dict) -> float:
    try:
        logger.debug("Calculating distance between two bounding boxes")

        center1 = np.mean([[v['x'], v['y']] for v in box1['vertices']], axis=0)
        center2 = np.mean([[v['x'], v['y']] for v in box2['vertices']], axis=0)

        distance = np.linalg.norm(center1 - center2)

        logger.debug(f"Calculated distance: {distance:.2f}")
        return distance

    except KeyError as e:
        logger.error(f"KeyError in calculate_distance: {str(e)}. Ensure both boxes have 'vertices' with 'x' and 'y' coordinates.", exc_info=True)
        return float('inf')  # Return infinity as a fallback for invalid boxes
    except Exception as e:
        logger.error(f"Error in calculate_distance: {str(e)}", exc_info=True)
        return float('inf')  # Return infinity as a fallback for any other errors
########################################################

## Removes sections where a brand has been over interpolated
########################################################
async def remove_over_interpolated_sections(brand_results: List[Dict], final_brand_appearances: Dict[str, Dict[int, List[Dict]]]) -> Tuple[List[Dict], Dict[str, Dict[int, List[Dict]]]]:
    try:
        logger.debug("Starting removal of over-interpolated sections")

        # Get unique brand names
        brand_names = set(brand['text'] for frame in brand_results for brand in frame['detected_brands'])
        logger.debug(f"Found {len(brand_names)} unique brands")

        async def process_brand(brand_name):
            logger.debug(f"Processing brand: {brand_name}")
            interpolated_sequences = await find_interpolated_sequences(brand_results, brand_name)
            logger.debug(f"Found {len(interpolated_sequences)} interpolated sequences for {brand_name}")

            for start, end in interpolated_sequences:
                if end - start + 1 > settings.INTERPOLATION_LIMIT:
                    logger.debug(f"Removing over-interpolated section for {brand_name} from frame {start} to {end}")
                    
                    # Remove all interpolated brands in this sequence
                    for frame in brand_results[start-brand_results[0]['frame_number']:end-brand_results[0]['frame_number']+1]:
                        frame['detected_brands'] = [b for b in frame['detected_brands'] if b['text'] != brand_name or not b.get('is_interpolated', False)]
                    
                    # Update final_brand_appearances
                    if brand_name in final_brand_appearances:
                        for frame_num in range(start, end + 1):
                            if frame_num in final_brand_appearances[brand_name]:
                                final_brand_appearances[brand_name][frame_num] = [
                                    b for b in final_brand_appearances[brand_name][frame_num]
                                    if not b.get('is_interpolated', False)
                                ]
                                if not final_brand_appearances[brand_name][frame_num]:
                                    del final_brand_appearances[brand_name][frame_num]
                                    logger.debug(f"Removed empty frame {frame_num} for {brand_name}")

            return brand_name

        # Process brands concurrently
        await asyncio.gather(*[process_brand(brand_name) for brand_name in brand_names])

        # Remove empty brand entries
        brands_to_remove = [k for k, v in final_brand_appearances.items() if not v]
        for brand in brands_to_remove:
            del final_brand_appearances[brand]

        logger.debug(f"Finished processing. {len(final_brand_appearances)} brands remain after removal")

        return brand_results, final_brand_appearances

    except Exception as e:
        logger.error(f"Error in remove_over_interpolated_sections: {str(e)}", exc_info=True)
        return brand_results, final_brand_appearances  # Return original data in case of error
########################################################

## Find sequences where a brand has been interpolated too many times
########################################################
async def find_interpolated_sequences(brand_results: List[Dict], brand_name: str) -> List[Tuple[int, int]]:
    try:
        logger.debug(f"Finding interpolated sequences for brand: {brand_name}")
        sequences = []
        start = None
        total_frames = len(brand_results)

        async def process_frame_chunk(chunk):
            nonlocal start
            local_sequences = []
            local_start = start

            for frame in chunk:
                interpolated_brands = [b for b in frame['detected_brands'] if b['text'] == brand_name and b.get('is_interpolated', False)]
                
                if interpolated_brands:
                    if local_start is None:
                        local_start = frame['frame_number']
                elif local_start is not None:
                    local_sequences.append((local_start, frame['frame_number'] - 1))
                    local_start = None

            return local_sequences, local_start

        # Process frames in chunks to reduce the number of tasks
        chunk_size = 1000
        chunks = [brand_results[i:i + chunk_size] for i in range(0, len(brand_results), chunk_size)]

        for i, chunk in enumerate(chunks):
            chunk_sequences, chunk_start = await process_frame_chunk(chunk)
            sequences.extend(chunk_sequences)
            start = chunk_start

            logger.debug(f"Processed chunk {i+1}/{len(chunks)} ({(i+1)*chunk_size}/{total_frames} frames)")

        if start is not None:
            sequences.append((start, brand_results[-1]['frame_number']))
            logger.debug(f"Ended final interpolated sequence: {start} to {brand_results[-1]['frame_number']}")

        logger.debug(f"Found {len(sequences)} interpolated sequences for brand {brand_name}")
        return sequences

    except Exception as e:
        logger.error(f"Error in find_interpolated_sequences for brand {brand_name}: {str(e)}", exc_info=True)
        return []  # Return an empty list in case of error
########################################################
## Detects if a brand has been interpolated too many times consecutively
########################################################
async def is_over_interpolated(brand: Dict, brand_results: List[Dict], current_frame: int) -> bool:
    try:
        logger.debug(f"Checking if brand '{brand['text']}' is over-interpolated at frame {current_frame}")

        if not brand.get('is_interpolated', False):
            logger.debug(f"Brand '{brand['text']}' is not interpolated at frame {current_frame}")
            return False

        brand_name = brand['text']
        sequences = await find_interpolated_sequences(brand_results, brand_name)

        for start, end in sequences:
            if start <= current_frame <= end:
                sequence_length = end - start + 1
                if sequence_length > settings.INTERPOLATION_LIMIT:
                    logger.info(f"Brand '{brand_name}' is over-interpolated at frame {current_frame}. "
                                        f"Sequence length: {sequence_length}, Limit: {settings.INTERPOLATION_LIMIT}")
                    return True
                else:
                    logger.debug(f"Brand '{brand_name}' is within interpolation limits at frame {current_frame}. "
                                            f"Sequence length: {sequence_length}, Limit: {settings.INTERPOLATION_LIMIT}")

        logger.debug(f"Brand '{brand_name}' is not over-interpolated at frame {current_frame}")
        return False

    except Exception as e:
        logger.error(f"Error in is_over_interpolated for brand '{brand['text']}' at frame {current_frame}: {str(e)}", exc_info=True)
        return False  # Assume not over-interpolated in case of error
########################################################