import re
import enchant
import asyncio
from typing import List, Dict, Set, Tuple
from thefuzz import fuzz
from core.config import settings
from services.ocr_processing import ocr_brand_matching

## Filter brand results from OCR
########################################################
async def filter_brand_results(vlogger, brand_results: List[Dict], brand_appearances: Dict[str, Set[int]], fps: float) -> List[Dict]:
    @vlogger.log_performance
    async def _filter_brand_results():
        try:
            vlogger.logger.info("Starting to filter brand results")

            min_frames = int(fps * settings.MIN_BRAND_TIME)
            vlogger.logger.debug(f"Minimum frames for a valid brand: {min_frames}")

            valid_brands = {brand for brand, appearances in brand_appearances.items() if len(appearances) >= min_frames}
            vlogger.logger.info(f"Found {len(valid_brands)} valid brands out of {len(brand_appearances)} total brands")

            async def process_frame(frame):
                filtered_brands = [brand for brand in frame['detected_brands'] if brand['text'] in valid_brands]
                return {
                    "frame_number": frame['frame_number'],
                    "detected_brands": filtered_brands
                }

            filtered_results = await asyncio.gather(*[process_frame(frame) for frame in brand_results])

            total_brands_before = sum(len(frame['detected_brands']) for frame in brand_results)
            total_brands_after = sum(len(frame['detected_brands']) for frame in filtered_results)

            vlogger.logger.info(f"Filtering complete. Brands reduced from {total_brands_before} to {total_brands_after}")
            return filtered_results

        except Exception as e:
            vlogger.logger.error(f"Error in filter_brand_results: {str(e)}", exc_info=True)
            return brand_results  # Return original results in case of error

    return await _filter_brand_results()
########################################################

## Clean raw OCR data
########################################################
async def clean_and_consolidate_ocr_data(vlogger, raw_ocr_results: List[Dict], video_dimensions: Tuple[int, int]) -> List[Dict]:
    @vlogger.log_performance
    async def _clean_and_consolidate_ocr_data():
        try:
            vlogger.logger.info("Starting OCR data cleaning and consolidation")
            d = enchant.Dict("en_US")
            video_width, video_height = video_dimensions

            def is_valid_word(word: str) -> bool:
                return d.check(word.lower())

            def preprocess_ocr_text(text: str) -> str:
                return text.lower()

            async def clean_detection(detection: Dict) -> Dict:
                if detection['Type'] != 'WORD':
                    return None
                
                original_text = detection['DetectedText']
                confidence = detection['Confidence']
                
                vlogger.logger.debug(f"Cleaning detection: {original_text} (confidence: {confidence})")
                
                if confidence > settings.MAX_CLEANING_CONFIDENCE:
                    final_text = original_text
                    vlogger.logger.debug(f"Skipping cleaning due to high confidence: {confidence}")
                else:
                    preprocessed_text = preprocess_ocr_text(original_text)
                    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', preprocessed_text)
                    
                    if len(cleaned_text) <= 2:
                        vlogger.logger.debug(f"Skipping detection due to short length: {cleaned_text}")
                        return None
                    
                    words = cleaned_text.split()
                    corrected_words = []
                    
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
                                vlogger.logger.debug(f"Corrected word: {word} -> {corrected_words[-1]}")
                            else:
                                corrected_words.append(word)
                    
                    final_text = ' '.join(corrected_words)
                    
                    # Check if all words in final_text are valid
                    if final_text.lower() != original_text.lower() and not all(is_valid_word(w) for w in final_text.split()):
                        final_text = original_text
                        vlogger.logger.debug(f"Reverted to original text: {original_text}")

                brand_match, brand_score = await asyncio.to_thread(ocr_brand_matching.fuzzy_match_brand, vlogger, final_text, min_score=85)
                
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

            async def process_frame(frame):
                vlogger.logger.debug(f"Processing frame {frame['frame_number']}")
                text_detections = frame['rekognition_response']['TextDetections']
                cleaned_detections = await asyncio.gather(*[clean_detection(det) for det in text_detections if det['Type'] == 'WORD'])
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
                                if await should_merge_bounding_boxes(vlogger, merged_detection['bounding_box'], detections[j]['bounding_box'], video_width, video_height):
                                    merged_box = await merge_bounding_boxes(vlogger, merged_detection['bounding_box'], detections[j]['bounding_box'])
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
                                    vlogger.logger.debug(f"Merged detections for brand: {brand}")
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
                    "text_model_version": frame['rekognition_response']['TextModelVersion'],
                    "average_confidence": avg_confidence
                }

            cleaned_results = await asyncio.gather(*[process_frame(frame) for frame in raw_ocr_results])

            vlogger.logger.info(f"Completed OCR data cleaning and consolidation. Processed {len(cleaned_results)} frames.")
            return cleaned_results

        except Exception as e:
            vlogger.logger.error(f"Error in clean_and_consolidate_ocr_data: {str(e)}", exc_info=True)
            return raw_ocr_results  # Return original results in case of error

    return await _clean_and_consolidate_ocr_data()
########################################################

## Calculate if bounding boxes are overlapping
########################################################
async def calculate_overlap(vlogger, box1: Dict, box2: Dict) -> float:
    @vlogger.log_performance
    def _calculate_overlap():
        try:
            vlogger.logger.debug("Calculating overlap between two bounding boxes")

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
            vlogger.logger.debug(f"Calculated overlap area: {overlap_area}")

            return overlap_area

        except KeyError as e:
            vlogger.logger.error(f"KeyError in calculate_overlap: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
            return 0.0  # Return no overlap in case of error
        except Exception as e:
            vlogger.logger.error(f"Error in calculate_overlap: {str(e)}", exc_info=True)
            return 0.0  # Return no overlap in case of error

    return await _calculate_overlap()
########################################################

## Evaluate whether we should merge bounding boxes
########################################################
async def should_merge_bounding_boxes(vlogger, box1: Dict, box2: Dict, frame_width: int, frame_height: int) -> bool:
    try:
        vlogger.logger.debug("Evaluating if bounding boxes should be merged")

        def close_edges(edge1, edge2, frame_dimension):
            max_distance = settings.MAX_BOUNDING_BOX_MERGE_DISTANCE_PERCENT * frame_dimension
            return abs(edge1 - edge2) <= max_distance

        v1, v2 = box1['vertices'], box2['vertices']

        # Check for overlap
        overlap_area = await calculate_overlap(vlogger, box1, box2)
        box1_area = (max(v1[1]['x'], v1[2]['x']) - min(v1[0]['x'], v1[3]['x'])) * (max(v1[2]['y'], v1[3]['y']) - min(v1[0]['y'], v1[1]['y']))
        box2_area = (max(v2[1]['x'], v2[2]['x']) - min(v2[0]['x'], v2[3]['x'])) * (max(v2[2]['y'], v2[3]['y']) - min(v2[0]['y'], v2[1]['y']))
        min_area = min(box1_area, box2_area)

        vlogger.logger.debug(f"Overlap area: {overlap_area}, Box1 area: {box1_area}, Box2 area: {box2_area}")

        if overlap_area > 0 and overlap_area / min_area >= settings.MIN_OVERLAP_RATIO_FOR_MERGE:
            vlogger.logger.info("Boxes should be merged due to significant overlap")
            return True

        # Check if bottom of box1 is close to top of box2 or vice versa
        if close_edges(max(v1[2]['y'], v1[3]['y']), min(v2[0]['y'], v2[1]['y']), frame_height) or \
           close_edges(max(v2[2]['y'], v2[3]['y']), min(v1[0]['y'], v1[1]['y']), frame_height):
            vlogger.logger.info("Boxes should be merged due to close vertical edges")
            return True

        # Check if right of box1 is close to left of box2 or vice versa
        if close_edges(max(v1[1]['x'], v1[2]['x']), min(v2[0]['x'], v2[3]['x']), frame_width) or \
           close_edges(max(v2[1]['x'], v2[2]['x']), min(v1[0]['x'], v1[3]['x']), frame_width):
            vlogger.logger.info("Boxes should be merged due to close horizontal edges")
            return True

        vlogger.logger.debug("Boxes should not be merged")
        return False

    except KeyError as e:
        vlogger.logger.error(f"KeyError in should_merge_bounding_boxes: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
        return False  # Return False in case of error
    except Exception as e:
        vlogger.logger.error(f"Error in should_merge_bounding_boxes: {str(e)}", exc_info=True)
        return False  # Return False in case of error
########################################################

## Merge bounding boxes
########################################################
async def merge_bounding_boxes(vlogger, box1: Dict, box2: Dict) -> Dict:
    @vlogger.log_performance
    def _merge_bounding_boxes():
        try:
            vlogger.logger.debug("Merging two bounding boxes")

            v1, v2 = box1['vertices'], box2['vertices']

            merged_box = {
                'vertices': [
                    {'x': min(v1[0]['x'], v2[0]['x']), 'y': min(v1[0]['y'], v2[0]['y'])},
                    {'x': max(v1[1]['x'], v2[1]['x']), 'y': min(v1[1]['y'], v2[1]['y'])},
                    {'x': max(v1[2]['x'], v2[2]['x']), 'y': max(v1[2]['y'], v2[2]['y'])},
                    {'x': min(v1[3]['x'], v2[3]['x']), 'y': max(v1[3]['y'], v2[3]['y'])}
                ]
            }

            vlogger.logger.info(f"Merged bounding box: {merged_box}")
            return merged_box

        except KeyError as e:
            vlogger.logger.error(f"KeyError in merge_bounding_boxes: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
            return box1  # Return first box in case of error
        except Exception as e:
            vlogger.logger.error(f"Error in merge_bounding_boxes: {str(e)}", exc_info=True)
            return box1  # Return first box in case of error

    return await _merge_bounding_boxes()
########################################################