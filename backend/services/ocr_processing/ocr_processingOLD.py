import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Callable, TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import StatusTracker

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
