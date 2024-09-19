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

    def clean_detection(detection: Dict) -> Dict:
        original_text = detection['DetectedText']
        confidence = detection['Confidence']
        
        # If confidence is high, skip intensive cleaning
        if confidence > settings.MAX_CLEANING_CONFIDENCE:
            cleaned_text = original_text
            final_text = original_text
        else:
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
            "cleaned_text": cleaned_text,
            "brand_match": brand_match,
            "brand_score": brand_score,
            "bounding_box": {"vertices": vertices},
            "type": detection['Type']
        }

    cleaned_results = []
    for frame in raw_ocr_results:
        text_detections = frame['rekognition_response']['TextDetections']
        cleaned_detections = [clean_detection(det) for det in text_detections]
        cleaned_detections = [det for det in cleaned_detections if det is not None]
        
        avg_confidence = sum(det['confidence'] for det in cleaned_detections) / len(cleaned_detections) if cleaned_detections else 0
        
        cleaned_frame = {
            "frame_number": frame['frame_number'],
            "full_text": ' '.join([det['text'] for det in cleaned_detections]),
            "cleaned_detections": cleaned_detections,  # This is the key change
            "text_model_version": frame['rekognition_response']['TextModelVersion'],
            "average_confidence": avg_confidence
        }
        cleaned_results.append(cleaned_frame)
    
    return cleaned_results
########################################################