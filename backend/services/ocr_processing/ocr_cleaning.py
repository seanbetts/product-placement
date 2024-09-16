import re
import enchant
from typing import List, Dict, Set
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
def clean_and_consolidate_ocr_data(ocr_results: List[Dict]) -> List[Dict]:
    d = enchant.Dict("en_US")

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
        brand_match, brand_score = ocr_brand_matching.fuzzy_match_brand(final_text, min_score=85)
        
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
########################################################