import re
import enchant
from typing import List, Dict, Set, Tuple
from thefuzz import fuzz
from core.config import settings
from services.ocr_processing import ocr_brand_matching

## Filter brand results from OCR
########################################################
def filter_brand_results(brand_results: List[Dict], brand_appearances: Dict[str, Set[int]], fps: float) -> List[Dict]:
    min_frames = int(fps * settings.MIN_BRAND_TIME)
    valid_brands = {brand for brand, appearances in brand_appearances.items() if len(appearances) >= min_frames}
    
    filtered_results = []
    for frame in brand_results:
        filtered_brands = [brand for brand in frame['detected_brands'] if brand['text'] in valid_brands]
        filtered_results.append({
            "frame_number": frame['frame_number'],
            "detected_brands": filtered_brands
        })
    
    return filtered_results
########################################################

## Clean raw OCR data
########################################################
def clean_and_consolidate_ocr_data(raw_ocr_results: List[Dict], video_dimensions: Tuple[int, int]) -> List[Dict]:
    d = enchant.Dict("en_US")
    video_width, video_height = video_dimensions

    def is_valid_word(word: str) -> bool:
        return d.check(word.lower())

    def preprocess_ocr_text(text: str) -> str:
        cleaned_text = text.lower()
        return cleaned_text

    def clean_detection(detection: Dict) -> Dict:
        if detection['Type'] != 'WORD':
            return None
        
        original_text = detection['DetectedText']
        confidence = detection['Confidence']
        
        # If confidence is high, skip cleaning
        if confidence > settings.MAX_CLEANING_CONFIDENCE:
            final_text = original_text
        else:
            preprocessed_text = preprocess_ocr_text(original_text)
            
            # Remove non-alphanumeric characters, preserving spaces
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', preprocessed_text)
            
            # Skip if the cleaned text is empty or too short
            if len(cleaned_text) <= 2:
                return None
            
            words = cleaned_text.split()
            corrected_words = []
            
            for word in words:
                if is_valid_word(word):
                    corrected_words.append(word)
                else:
                    suggestions = d.suggest(word)
                    if suggestions and fuzz.ratio(word.lower(), suggestions[0].lower()) > settings.MIN_WORD_MATCH:
                        # Preserve original capitalization
                        if word.isupper():
                            corrected_words.append(suggestions[0].upper())
                        elif word.istitle():
                            corrected_words.append(suggestions[0].title())
                        else:
                            corrected_words.append(suggestions[0].lower())
                    else:
                        corrected_words.append(word)  # Keep original if no good suggestion
            
            final_text = ' '.join(corrected_words)
            
            # If the final text is not valid and different from the original, revert to original
            if final_text.lower() != original_text.lower() and not all(is_valid_word(w) for w in final_text.split()):
                final_text = original_text

        # Brand matching
        brand_match, brand_score = ocr_brand_matching.fuzzy_match_brand(final_text, min_score=85)

        # Convert Polygon data to pixel coordinates
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
            "cleaned_text": final_text,  # Now this will be either a valid correction or the original text
            "brand_match": brand_match,
            "brand_score": brand_score,
            "bounding_box": {"vertices": vertices},
            "type": detection['Type']
        }

    cleaned_results = []
    for frame in raw_ocr_results:
        text_detections = frame['rekognition_response']['TextDetections']
        cleaned_detections = [clean_detection(det) for det in text_detections if det['Type'] == 'WORD']
        cleaned_detections = [det for det in cleaned_detections if det is not None]
        
        # Group detections by brand
        brand_groups = {}
        for det in cleaned_detections:
            brand = det['brand_match'] if det['brand_match'] else det['text']
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(det)
        
        # Merge bounding boxes for each brand group
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
                        if should_merge_bounding_boxes(merged_detection['bounding_box'], detections[j]['bounding_box'], video_width, video_height):
                            merged_box = merge_bounding_boxes(merged_detection['bounding_box'], detections[j]['bounding_box'])
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
                    new_detections.append(merged_detection)
                if not merged:
                    break
                detections = new_detections
            merged_detections.extend(detections)
        
        avg_confidence = sum(det['confidence'] for det in merged_detections) / len(merged_detections) if merged_detections else 0
        
        cleaned_frame = {
            "frame_number": frame['frame_number'],
            "full_text": ' '.join([det['text'] for det in merged_detections]),
            "cleaned_detections": merged_detections,
            "text_model_version": frame['rekognition_response']['TextModelVersion'],
            "average_confidence": avg_confidence
        }
        cleaned_results.append(cleaned_frame)

    return cleaned_results
########################################################

## Calculate if bounding boxes are overlapping
########################################################
def calculate_overlap(box1: Dict, box2: Dict) -> float:
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

    return x_overlap * y_overlap
########################################################

## Evaluate whether we should merge bounding boxes
########################################################
def should_merge_bounding_boxes(box1: Dict, box2: Dict, frame_width: int, frame_height: int) -> bool:
    def close_edges(edge1, edge2, frame_dimension):
        max_distance = settings.MAX_BOUNDING_BOX_MERGE_DISTANCE_PERCENT * frame_dimension
        return abs(edge1 - edge2) <= max_distance

    v1, v2 = box1['vertices'], box2['vertices']
    
    # Check for overlap
    overlap_area = calculate_overlap(box1, box2)
    box1_area = (max(v1[1]['x'], v1[2]['x']) - min(v1[0]['x'], v1[3]['x'])) * (max(v1[2]['y'], v1[3]['y']) - min(v1[0]['y'], v1[1]['y']))
    box2_area = (max(v2[1]['x'], v2[2]['x']) - min(v2[0]['x'], v2[3]['x'])) * (max(v2[2]['y'], v2[3]['y']) - min(v2[0]['y'], v2[1]['y']))
    min_area = min(box1_area, box2_area)
    
    if overlap_area > 0 and overlap_area / min_area >= settings.MIN_OVERLAP_RATIO_FOR_MERGE:
        return True
    
    # Check if bottom of box1 is close to top of box2 or vice versa
    if close_edges(max(v1[2]['y'], v1[3]['y']), min(v2[0]['y'], v2[1]['y']), frame_height) or \
       close_edges(max(v2[2]['y'], v2[3]['y']), min(v1[0]['y'], v1[1]['y']), frame_height):
        return True
    
    # Check if right of box1 is close to left of box2 or vice versa
    if close_edges(max(v1[1]['x'], v1[2]['x']), min(v2[0]['x'], v2[3]['x']), frame_width) or \
       close_edges(max(v2[1]['x'], v2[2]['x']), min(v1[0]['x'], v1[3]['x']), frame_width):
        return True
    
    return False
########################################################

## Merge bounding boxes
########################################################
def merge_bounding_boxes(box1: Dict, box2: Dict) -> Dict:
    v1, v2 = box1['vertices'], box2['vertices']
    
    return {
        'vertices': [
            {'x': min(v1[0]['x'], v2[0]['x']), 'y': min(v1[0]['y'], v2[0]['y'])},
            {'x': max(v1[1]['x'], v2[1]['x']), 'y': min(v1[1]['y'], v2[1]['y'])},
            {'x': max(v1[2]['x'], v2[2]['x']), 'y': max(v1[2]['y'], v2[2]['y'])},
            {'x': min(v1[3]['x'], v2[3]['x']), 'y': max(v1[3]['y'], v2[3]['y'])}
        ]
    }
########################################################