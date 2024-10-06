import re
import enchant
import asyncio
from typing import List, Dict, Set, Tuple
from thefuzz import fuzz
from core.config import settings
from core.logging import logger
from services.ocr_processing import ocr_brand_matching

## Clean raw OCR data
########################################################
async def clean_and_consolidate_ocr_data(raw_ocr_results: List[Dict], video_dimensions: Tuple[int, int]) -> List[Dict]:
    try:
        logger.info("Video Processing - Brand Detection - Step 3.2.2: Starting OCR data cleaning and consolidation")

        video_width, video_height = video_dimensions

        # Load the brand database from settings
        brand_database = settings.BRAND_DATABASE

        cleaned_results = await asyncio.gather(*[process_frame(frame, brand_database, video_width, video_height) for frame in raw_ocr_results])

        logger.info(f"Video Processing - Brand Detection - Step 3.2.3: Completed OCR data cleaning and consolidation. Processed {len(cleaned_results)} frames.")
        return cleaned_results

    except Exception as e:
        logger.error(f"OCR Data Processing: Error in clean_and_consolidate_ocr_data: {str(e)}", exc_info=True)
        return raw_ocr_results  # Return original results in case of error
########################################################

## Process each frame
########################################################
async def process_frame(frame, brand_database, video_width, video_height):
    logger.debug(f"Processing frame {frame['frame_number']}")
    text_detections = frame['rekognition_response'].get('TextDetections', [])

    # Apply the new brand-aware filtering using 'DetectedText'
    filtered_detections = await filter_detections_by_brand(
        detections=text_detections,
        brand_database=brand_database,
        video_width=video_width,
        video_height=video_height
    )

    cleaned_detections = await asyncio.gather(*[clean_detection(det, video_width, video_height) for det in filtered_detections])
    cleaned_detections = [det for det in cleaned_detections if det is not None]

    brand_groups = {}
    for det in cleaned_detections:
        brand = det['brand_match'] if det['brand_match'] else det['text']
        if brand not in brand_groups:
            brand_groups[brand] = []
        brand_groups[brand].append(det)

    merged_detections = []
    for brand, detections in brand_groups.items():
        while len(detections) > 1:
            merged = False
            new_detections = []
            skip_indices = set()
            for i in range(len(detections)):
                if i in skip_indices:
                    continue
                merged_detection = detections[i]
                for j in range(i + 1, len(detections)):
                    if j in skip_indices:
                        continue
                    box1 = {
                        **merged_detection['bounding_box'],
                        "text": merged_detection['text'],
                        "cleaned_text": merged_detection['cleaned_text'],
                        "confidence": merged_detection['confidence']
                    }
                    box2 = {
                        **detections[j]['bounding_box'],
                        "text": detections[j]['text'],
                        "cleaned_text": detections[j]['cleaned_text'],
                        "confidence": detections[j]['confidence']
                    }
                    if await should_merge_bounding_boxes(box1, box2, video_width, video_height):
                        merged_box = await merge_bounding_boxes(merged_detection['bounding_box'], detections[j]['bounding_box'])
                        merged_detection = {
                            **merged_detection,
                            'original_text': f"{merged_detection['original_text']} {detections[j]['original_text']}",
                            'text': brand,
                            'cleaned_text': f"{merged_detection['cleaned_text']} {detections[j]['cleaned_text']}",
                            'confidence': (merged_detection['confidence'] + detections[j]['confidence']) / 2,
                            'bounding_box': merged_box,
                            'merged': True
                        }
                        skip_indices.add(j)
                        merged = True
                        logger.debug(f"Merged detections for brand: {brand}")
                new_detections.append(merged_detection)
            if not merged:
                break
            detections = new_detections
        merged_detections.extend(detections)

    avg_confidence = sum(det['confidence'] for det in merged_detections) / len(merged_detections) if merged_detections else 0

    return {
        "frame_number": frame['frame_number'],
        "full_text": ' '.join([det['text'] for det in merged_detections]),
        "cleaned_detections": merged_detections,
        "text_model_version": frame['rekognition_response'].get('TextModelVersion', ''),
        "average_confidence": avg_confidence
    }
########################################################

## Replace LINE detections that contain more than just the brand name with WORD detections
########################################################
async def filter_detections_by_brand(
    detections: List[Dict],
    brand_database: Dict[str, Dict],
    video_width: int,
    video_height: int
) -> List[Dict]:
    """
    Filters and replaces 'LINE' detections containing brand names with corresponding 'WORD' detections.
    
    Args:
        detections (List[Dict]): List of OCR detections for a frame.
        brand_database (Dict[str, Dict]): Brand database with brand names and their variations.
        video_width (int): Width of the video frame.
        video_height (int): Height of the video frame.
    
    Returns:
        List[Dict]: Filtered and possibly modified list of detections.
    """
    # Compile a list of all brand names and their variations
    brand_variations = {}
    for brand, details in brand_database.items():
        variations = [brand.lower()] + [v.lower() for v in details.get('variations', [])]
        brand_variations[brand.lower()] = variations

    # Separate 'LINE' and 'WORD' detections
    line_detections = [det for det in detections if det.get('Type') == 'LINE']
    word_detections = [det for det in detections if det.get('Type') == 'WORD']

    # Initialize a set to keep track of 'WORD' detections to retain
    retained_word_indices = set()

    # Initialize a list to store the final detections
    final_detections = []

    for line_det in line_detections:
        detected_text = line_det.get('DetectedText', '')
        if not detected_text:
            logger.debug("Skipping detection with missing 'DetectedText'.")
            continue  # Skip detections without 'DetectedText'

        line_text = detected_text.lower()
        matched_brands = []

        # Check for brand presence in the line text
        for brand, variations in brand_variations.items():
            for variation in variations:
                # Use word boundaries to ensure exact matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, line_text):
                    matched_brands.append((brand, variation))
                    break  # Avoid multiple matches for the same brand

        if not matched_brands:
            # No brand found in this 'LINE' detection; retain it as is
            final_detections.append(line_det)
            continue

        # Determine if the 'LINE' contains only the brand name(s)
        # Remove all matched brand variations from the line text
        modified_text = line_text
        for _, variation in matched_brands:
            modified_text = re.sub(r'\b' + re.escape(variation) + r'\b', '', modified_text)

        # Check if the remaining text has non-whitespace characters
        if modified_text.strip():
            # The 'LINE' contains additional words besides the brand name(s)
            # Replace with corresponding 'WORD' detections for the brand name(s)
            for brand, variation in matched_brands:
                # Split the brand variation into individual words
                brand_words = variation.split()
                for word in brand_words:
                    # Find 'WORD' detections that match this word
                    for idx, word_det in enumerate(word_detections):
                        if idx in retained_word_indices:
                            continue  # Skip already retained detections
                        word_text = word_det.get('DetectedText', '').lower()
                        if word_text == word.lower():
                            final_detections.append(word_det)
                            retained_word_indices.add(idx)
                            break  # Move to the next word
            # Optionally, you can log or handle the replacement here
            logger.debug(f"Replaced 'LINE' detection '{detected_text}' with 'WORD' detections for brands.")
        else:
            # The 'LINE' contains only the brand name(s); retain it
            final_detections.append(line_det)

    # Add the remaining 'WORD' detections that were not part of any replacement
    for idx, word_det in enumerate(word_detections):
        if idx not in retained_word_indices:
            final_detections.append(word_det)

    return final_detections
########################################################

## Clean detection
########################################################
async def clean_detection(detection: Dict, video_width: int, video_height: int) -> Dict:
    if detection['Type'] != settings.OCR_TYPE:
        return None

    original_text = detection['DetectedText']
    confidence = detection['Confidence']

    d = enchant.Dict("en_US")
    
    logger.debug(f"Cleaning detection: {original_text} (confidence: {confidence})")
    
    if confidence > settings.MAX_CLEANING_CONFIDENCE:
        final_text = original_text
        logger.debug(f"Skipping cleaning due to high confidence: {confidence}")
    else:
        preprocessed_text = original_text.lower()
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', preprocessed_text)
        
        if len(cleaned_text) <= 2:
            logger.debug(f"Skipping detection due to short length: {cleaned_text}")
            return None
        
        words = cleaned_text.split()
        corrected_words = []

        def is_valid_word(word: str) -> bool:
            return d.check(word.lower())
        
        for word in words:
            if is_valid_word(word):
                corrected_words.append(word)
            else:
                suggestions = d.suggest(word)
                if suggestions and fuzz.ratio(word.lower(), suggestions[0].lower()) > settings.MIN_WORD_MATCH:
                    if word.isupper():
                        corrected_words.append(suggestions[0].upper())
                    elif word.istitle():
                        corrected_words.append(suggestions[0].title())
                    else:
                        corrected_words.append(suggestions[0].lower())
                    logger.debug(f"Corrected word: {word} -> {corrected_words[-1]}")
                else:
                    corrected_words.append(word)
        
        final_text = ' '.join(corrected_words)
        
        # Check if all words in final_text are valid
        if final_text.lower() != original_text.lower() and not all(is_valid_word(w) for w in final_text.split()):
            final_text = original_text
            logger.debug(f"Reverted to original text: {original_text}")

    brand_match, brand_score = await asyncio.to_thread(ocr_brand_matching.fuzzy_match_brand, final_text, min_score=85)
    
    vertices = [
        {
            "x": int(point['X'] * video_width),
            "y": int(point['Y'] * video_height)
        }
        for point in detection['Geometry']['Polygon']
    ]

    return {
        "original_text": original_text,
        "confidence": confidence,
        "text": brand_match if brand_match else final_text,
        "cleaned_text": final_text,
        "brand_match": brand_match,
        "brand_score": brand_score,
        "bounding_box": {"vertices": vertices},
        "type": detection['Type']
    }
########################################################

## Calculate if bounding boxes are overlapping
########################################################
async def calculate_overlap(box1: Dict, box2: Dict) -> float:
    try:
        logger.debug("Calculating overlap between two bounding boxes")

        def calculate_overlap_1d(min1, max1, min2, max2):
            return max(0, min(max1, max2) - max(min1, min2))

        x_overlap = calculate_overlap_1d(
            min(box1['vertices'][0]['x'], box1['vertices'][3]['x']),
            max(box1['vertices'][1]['x'], box1['vertices'][2]['x']),
            min(box2['vertices'][0]['x'], box2['vertices'][3]['x']),
            max(box2['vertices'][1]['x'], box2['vertices'][2]['x'])
        )

        y_overlap = calculate_overlap_1d(
            min(box1['vertices'][0]['y'], box1['vertices'][1]['y']),
            max(box1['vertices'][2]['y'], box1['vertices'][3]['y']),
            min(box2['vertices'][0]['y'], box2['vertices'][1]['y']),
            max(box2['vertices'][2]['y'], box2['vertices'][3]['y'])
        )

        overlap_area = x_overlap * y_overlap
        logger.debug(f"Calculated overlap area: {overlap_area}")

        return overlap_area

    except KeyError as e:
        logger.error(f"OCR Data Processing: KeyError in calculate_overlap: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
        return 0.0  # Return no overlap in case of error
    except Exception as e:
        logger.error(f"OCR Data Processing: Error in calculate_overlap: {str(e)}", exc_info=True)
        return 0.0  # Return no overlap in case of error
########################################################

## Evaluate whether we should merge bounding boxes
########################################################
async def should_merge_bounding_boxes(box1: Dict, box2: Dict, frame_width: int, frame_height: int) -> bool:
    try:
        logger.debug("Evaluating if bounding boxes should be merged")

        def close_edges(edge1, edge2, frame_dimension):
            max_distance = settings.MAX_BOUNDING_BOX_MERGE_DISTANCE_PERCENT * frame_dimension
            return abs(edge1 - edge2) <= max_distance

        v1, v2 = box1.get('vertices', []), box2.get('vertices', [])

        # Ensure the vertices are provided
        if not v1 or not v2:
            logger.error("Vertices not provided in one or both bounding boxes")
            return False

        # Extract texts and confidences from boxes
        box1_text = box1.get('cleaned_text', '').lower()
        box2_text = box2.get('cleaned_text', '').lower()
        box1_full_text = box1.get('text', '').lower()
        box2_full_text = box2.get('text', '').lower()
        confidence1 = box1.get('confidence', 0.0)
        confidence2 = box2.get('confidence', 0.0)

        # Check if cleaned_text matches the full brand name in the text field
        if box1_text == box1_full_text and box2_text == box2_full_text:
            logger.debug("Boxes should not be merged as they contain the full brand name already")
            return False

        # Check if the combined cleaned_text of both boxes matches the text field of either box
        combined_text = f"{box1_text} {box2_text}".strip()
        if combined_text == box1_full_text or combined_text == box2_full_text:
            logger.debug("Combined cleaned text matches full brand name, proceeding to evaluate proximity and overlap")
        else:
            logger.debug("Boxes do not form the full brand name, should not be merged")
            return False

        # Check for overlap
        overlap_area = await calculate_overlap(box1, box2)
        box1_x_min = min(v1[0]['x'], v1[3]['x'])
        box1_x_max = max(v1[1]['x'], v1[2]['x'])
        box1_y_min = min(v1[0]['y'], v1[1]['y'])
        box1_y_max = max(v1[2]['y'], v1[3]['y'])
        box1_area = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)

        box2_x_min = min(v2[0]['x'], v2[3]['x'])
        box2_x_max = max(v2[1]['x'], v2[2]['x'])
        box2_y_min = min(v2[0]['y'], v2[1]['y'])
        box2_y_max = max(v2[2]['y'], v2[3]['y'])
        box2_area = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

        min_area = min(box1_area, box2_area)

        logger.debug(f"Overlap area: {overlap_area}, Box1 area: {box1_area}, Box2 area: {box2_area}")

        if overlap_area > 0 and overlap_area / min_area >= settings.MIN_OVERLAP_RATIO_FOR_MERGE:
            logger.debug("Boxes should be merged due to significant overlap")
            return True

        # Check if bottom of box1 is close to top of box2 or vice versa
        if close_edges(box1_y_max, box2_y_min, frame_height) or \
           close_edges(box2_y_max, box1_y_min, frame_height):
            logger.debug("Boxes should be merged due to close vertical edges")
            return True

        # Check if right of box1 is close to left of box2 or vice versa
        if close_edges(box1_x_max, box2_x_min, frame_width) or \
           close_edges(box2_x_max, box1_x_min, frame_width):
            logger.debug("Boxes should be merged due to close horizontal edges")
            return True

        logger.debug("Boxes should not be merged")
        return False

    except KeyError as e:
        logger.error(f"OCR Data Processing: KeyError in should_merge_bounding_boxes: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
        return False  # Return False in case of error
    except Exception as e:
        logger.error(f"OCR Data Processing: Error in should_merge_bounding_boxes: {str(e)}", exc_info=True)
        return False  # Return False in case of error
########################################################

## Merge bounding boxes
########################################################
async def merge_bounding_boxes(box1: Dict, box2: Dict) -> Dict:
    try:
        logger.debug("Merging two bounding boxes")

        v1, v2 = box1['vertices'], box2['vertices']

        merged_box = {
            'vertices': [
                {'x': min(v1[0]['x'], v2[0]['x']), 'y': min(v1[0]['y'], v2[0]['y'])},
                {'x': max(v1[1]['x'], v2[1]['x']), 'y': min(v1[1]['y'], v2[1]['y'])},
                {'x': max(v1[2]['x'], v2[2]['x']), 'y': max(v1[2]['y'], v2[2]['y'])},
                {'x': min(v1[3]['x'], v2[3]['x']), 'y': max(v1[3]['y'], v2[3]['y'])}
            ]
        }

        logger.debug(f"Merged bounding box: {merged_box}")
        return merged_box

    except KeyError as e:
        logger.error(f"OCR Data Processing: KeyError in merge_bounding_boxes: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
        return box1  # Return first box in case of error
    except Exception as e:
        logger.error(f"OCR Data Processing: Error in merge_bounding_boxes: {str(e)}", exc_info=True)
        return box1  # Return first box in case of error
########################################################