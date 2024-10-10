import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque, Counter, defaultdict
from thefuzz import fuzz
from core.config import settings
from core.logging import logger

## Data Classes
########################################################
@dataclass
class Frame:
    number: int
    width: int
    height: int
    ocr_data: Dict

@dataclass
class BrandInstance:
    brand: str
    brand_match_confidence: float
    original_detected_text: str
    original_detected_text_confidence: float
    cleaned_text: str
    bounding_box: Dict
    frame_number: int
    detection_type: str
    relative_size: float
    position: Dict[str, float]
    frame_timestamp: float
    fuzzy_match_term: Optional[str] = None
    fuzzy_match_score: Optional[float] = None
    is_interpolated: bool = False
    interpolation_confidence: Optional[float] = None
    interpolation_source_frames: Optional[Tuple[int, int]] = None
    merged_from: Optional[List[Dict]] = None

    def to_dict(self):
        return {
            "brand": self.brand,
            "brand_match_confidence": self.brand_match_confidence,
            "original_detected_text": self.original_detected_text,
            "original_detected_text_confidence": self.original_detected_text_confidence,
            "cleaned_text": self.cleaned_text,
            "bounding_box": self.bounding_box,
            "frame_number": self.frame_number,
            "detection_type": self.detection_type,
            "relative_size": self.relative_size,
            "position": self.position,
            "frame_timestamp": self.frame_timestamp,
            "fuzzy_match_term": self.fuzzy_match_term,
            "fuzzy_match_score": self.fuzzy_match_score,
            "is_interpolated": self.is_interpolated,
            "interpolation_confidence": self.interpolation_confidence,
            "interpolation_source_frames": self.interpolation_source_frames,
            "merged_from": self.merged_from
        }

@dataclass
class DetectionResult:
    frame_number: int
    detected_brands: List[BrandInstance]

    def to_dict(self):
        return {
            "frame_number": self.frame_number,
            "detected_brands": [detected_brand.to_dict() for detected_brand in self.detected_brands]
        }
########################################################

## BrandDetector Class
########################################################
class BrandDetector:
    def __init__(self, fps: float, frame_width: int, frame_height: int):
        self.brand_database = settings.BRAND_DATABASE
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        window_size = int(self.fps * settings.FRAME_WINDOW)
        self.brand_buffer = deque(maxlen=window_size)
        self.all_detections = defaultdict(list)  
        self.consistent_brands = set()
        self.brand_appearances = defaultdict(set)
        self.accumulated_results = {}
        self.all_brand_detections = []
        self.current_frame_number = 0
        self.video_resolution = None

    async def process_frame(self, frame: Frame) -> DetectionResult:
        self.current_frame_number = frame.number
        self.video_resolution = (frame.width, frame.height)

        logger.debug(f"Processing frame {frame.number}")

        text_detections = frame.ocr_data['rekognition_response'].get('TextDetections', [])
        logger.debug(f"Number of text detections: {len(text_detections)}")

        cleaned_detections = [await self.clean_detection(det) for det in text_detections]
        cleaned_detections = [det for det in cleaned_detections if det is not None]
        logger.debug(f"Cleaned detections: {len(cleaned_detections)} items")

        matched_brands = []
        for det in cleaned_detections:
            brand = await self.match_brand(det)
            if brand:
                matched_brands.append(brand)
                logger.debug(f"Frame {frame.number}: Matched brand: {brand.brand}, Score: {brand.brand_match_confidence}")

        logger.debug(f"Frame {frame.number}: Matched brands: {len(matched_brands)}")

        # Group detections by brand
        brand_groups = {}
        for brand in matched_brands:
            if brand.brand not in brand_groups:
                brand_groups[brand.brand] = []
            brand_groups[brand.brand].append(brand)

        merged_brands = []
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
                        if await self.should_merge_bounding_boxes(merged_detection.bounding_box, detections[j].bounding_box, frame.width, frame.height):
                            merged_box, merged_from = await self.merge_bounding_boxes(merged_detection.bounding_box, detections[j].bounding_box)
                            relative_size = await self.calculate_relative_size(merged_box)
                            position = await self.calculate_position(merged_box)
                            merged_detection = BrandInstance(
                                brand=brand,
                                brand_match_confidence=(merged_detection.brand_match_confidence + detections[j].brand_match_confidence) / 2,
                                original_detected_text=f"{merged_detection.original_detected_text} {detections[j].original_detected_text}",
                                original_detected_text_confidence=(merged_detection.original_detected_text_confidence + detections[j].original_detected_text_confidence) / 2,
                                cleaned_text=f"{merged_detection.cleaned_text} {detections[j].cleaned_text}",
                                bounding_box=merged_box,
                                frame_number=frame.number,
                                detection_type="MERGED",
                                relative_size=relative_size,
                                position=position,
                                frame_timestamp=frame.number / self.fps,
                                fuzzy_match_term=merged_detection.fuzzy_match_term,
                                fuzzy_match_score=merged_detection.fuzzy_match_score,
                                is_interpolated=False,
                                merged_from=merged_from
                            )
                            skip_indices.add(j)
                            merged = True
                    new_detections.append(merged_detection)
                if not merged:
                    break
                detections = new_detections
            merged_brands.extend(detections)

        logger.debug(f"Frame {frame.number}: Merged brands: {len(merged_brands)}")

        # Store all detections
        self.all_detections[frame.number] = merged_brands

        # Update brand_buffer with new detections
        for brand in merged_brands:
            self.brand_buffer.append(brand)

        # Implement sliding window for consistency check
        window_size = int(self.fps * settings.FRAME_WINDOW)
        current_frame = frame.number

        logger.debug(f"Frame {current_frame}: Brand buffer size: {len(self.brand_buffer)}")

        # Count occurrences of each brand in the buffer
        brand_counts = Counter(b.brand for b in self.brand_buffer)

        consistent_brands_this_frame = []
        for brand in merged_brands:
            occurrences = brand_counts[brand.brand]
            
            logger.debug(f"Frame {current_frame}: Brand {brand.brand} has {occurrences} occurrences in the last {window_size} frames")

            if occurrences >= settings.MIN_DETECTIONS:
                consistent_brands_this_frame.append(brand)
                self.consistent_brands.add(brand.brand)
                logger.debug(f"Frame {current_frame}: Brand {brand.brand} is consistent with {occurrences} occurrences in the last {window_size} frames")
            else:
                logger.debug(f"Frame {current_frame}: Brand {brand.brand} not consistent enough (only {occurrences} occurrences in the last {window_size} frames)")

        # Include all detections for consistent brands within the window
        all_consistent_brands = []
        window_start = max(0, current_frame - window_size + 1)
        for frame_num in range(window_start, current_frame + 1):
            for brand in self.all_detections.get(frame_num, []):
                if brand.brand in self.consistent_brands:
                    all_consistent_brands.append(brand)

        logger.debug(f"Frame {frame.number}: All consistent brands: {len(all_consistent_brands)}")

        interpolated_brands = await self.interpolate_brands(frame.number, all_consistent_brands)
        logger.debug(f"Frame {frame.number}: Brands after interpolation: {[b.brand for b in interpolated_brands]}")

        filtered_brands = await self.remove_over_interpolated_brands(interpolated_brands)
        logger.debug(f"Frame {frame.number}: Brands after removing over-interpolated: {[b.brand for b in filtered_brands]}")

        # Accumulate brand appearances instead of filtering
        for brand in filtered_brands:
            self.brand_appearances[brand.brand].add(frame.number)
        self.all_brand_detections.extend(filtered_brands)

        logger.debug(f"Frame {frame.number}: Accumulated brands: {[b.brand for b in filtered_brands]}")

        self.accumulated_results[frame.number] = filtered_brands

        return DetectionResult(frame.number, filtered_brands)

    async def clean_detection(self, detection: Dict) -> Optional[Dict]:
        try:
            logger.debug(f"Cleaning detection: {detection['Type']} - {detection['DetectedText']}")
            
            if detection['Type'] != settings.OCR_TYPE:
                return None

            confidence = detection['Confidence']
            
            # Add this check
            if confidence < settings.MINIMUM_OCR_CONFIDENCE:
                logger.debug(f"Skipping detection due to low confidence: {confidence}")
                return None

            original_text = detection['DetectedText']

            if confidence > settings.MAX_CLEANING_CONFIDENCE:
                cleaned_text = original_text
            else:
                cleaned_text = await self.clean_text(original_text)

            logger.debug(f"Frame {self.current_frame_number}: Cleaned text: '{original_text}' -> '{cleaned_text}'")

            if cleaned_text is None or len(cleaned_text.strip()) <= 2:
                logger.debug(f"Skipping detection due to short or None cleaned text: '{original_text}'")
                return None

            bounding_box = await self.convert_relative_bbox(detection['Geometry']['BoundingBox'])

            return {
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "frame_number": self.current_frame_number
            }
        except Exception as e:
            logger.error(f"Error in clean_detection: {str(e)}", exc_info=True)
            return None

    async def match_brand(self, detection: Dict) -> Optional[BrandInstance]:
        try:
            brand_match, brand_score = await self.fuzzy_match_brand(detection['cleaned_text'])
            
            logger.debug(f"Frame {self.current_frame_number}: Fuzzy match result: text='{detection['cleaned_text']}', match='{brand_match}', score={brand_score}")
            
            if brand_match and brand_score >= settings.MINIMUM_FUZZY_BRAND_MATCH_SCORE:
                # Combine OCR confidence and fuzzy match score
                combined_confidence = (brand_score + detection['confidence']) / 2
                
                logger.debug(f"Frame {self.current_frame_number}: Matched brand: {brand_match} with combined confidence {combined_confidence}")
                relative_size = await self.calculate_relative_size(detection['bounding_box'])
                position = await self.calculate_position(detection['bounding_box'])
                frame_timestamp = self.current_frame_number / self.fps
                
                return BrandInstance(
                    brand=brand_match,
                    brand_match_confidence=combined_confidence,
                    original_detected_text=detection['original_text'],
                    original_detected_text_confidence=detection['confidence'],
                    cleaned_text=detection['cleaned_text'],
                    bounding_box=detection['bounding_box'],
                    frame_number=self.current_frame_number,
                    detection_type="OCR",
                    relative_size=relative_size,
                    position=position,
                    frame_timestamp=frame_timestamp,
                    fuzzy_match_term=detection['cleaned_text'],
                    fuzzy_match_score=brand_score,
                    is_interpolated=False
                )
            else:
                logger.debug(f"Frame {self.current_frame_number}: No brand match for '{detection['cleaned_text']}' (score: {brand_score})")
            return None
        except Exception as e:
            logger.error(f"Error in match_brand: {str(e)}", exc_info=True)
            return None

    async def interpolate_brands(self, frame_number: int, consistent_brands: List[BrandInstance]) -> List[BrandInstance]:
        interpolated_brands = []
        
        current_frame_brands = [b for b in consistent_brands if b.frame_number == frame_number]
        
        # Add current frame brands to the result without interpolation
        interpolated_brands.extend(current_frame_brands)
        
        # Get all unique brands that have appeared in the video
        all_brands = set(brand.brand for brand in self.all_brand_detections)
        
        for brand_name in all_brands:
            if brand_name not in [b.brand for b in current_frame_brands]:
                interpolated_brand = await self.interpolate_brand(brand_name, frame_number)
                if interpolated_brand:
                    interpolated_brands.append(interpolated_brand)
        
        return interpolated_brands

    async def interpolate_brand(self, brand_name: str, frame_number: int) -> Optional[BrandInstance]:
        # Find the nearest previous and next instances
        prev_instance = next((b for b in reversed(self.all_brand_detections) if b.brand == brand_name and b.frame_number < frame_number), None)
        next_instance = next((b for b in self.all_brand_detections if b.brand == brand_name and b.frame_number > frame_number), None)

        if prev_instance and next_instance:
            frame_diff = next_instance.frame_number - prev_instance.frame_number
            max_interpolation_limit = int(self.fps * 5)  # Allow up to 5 seconds of interpolation
            
            if frame_diff <= max_interpolation_limit:
                return await self._interpolate_between(prev_instance, next_instance, frame_number)
        
        return None

    async def _interpolate_between(self, prev: BrandInstance, next: BrandInstance, frame_number: int) -> BrandInstance:
        t = (frame_number - prev.frame_number) / (next.frame_number - prev.frame_number)
        
        interpolated_box = {
            'vertices': [
                {k: int(prev.bounding_box['vertices'][i][k] +
                    t * (next.bounding_box['vertices'][i][k] - prev.bounding_box['vertices'][i][k]))
                for k in ['x', 'y']}
                for i in range(4)
            ]
        }

        confidence = prev.brand_match_confidence + t * (next.brand_match_confidence - prev.brand_match_confidence)
        relative_size = prev.relative_size + t * (next.relative_size - prev.relative_size)
        position = {
            "x": prev.position["x"] + t * (next.position["x"] - prev.position["x"]),
            "y": prev.position["y"] + t * (next.position["y"] - prev.position["y"])
        }
        frame_timestamp = frame_number / self.fps
        
        interpolation_confidence = 1 - (frame_number - prev.frame_number) / (next.frame_number - prev.frame_number)
        
        return BrandInstance(
            brand=prev.brand,
            brand_match_confidence=confidence * interpolation_confidence,
            original_detected_text="[INTERPOLATED]",
            original_detected_text_confidence=0,
            cleaned_text=prev.cleaned_text,
            bounding_box=interpolated_box,
            frame_number=frame_number,
            detection_type="INTERPOLATED",
            relative_size=relative_size,
            position=position,
            frame_timestamp=frame_timestamp,
            fuzzy_match_term=prev.fuzzy_match_term,
            fuzzy_match_score=prev.fuzzy_match_score,
            is_interpolated=True,
            interpolation_confidence=interpolation_confidence,
            interpolation_source_frames=(prev.frame_number, next.frame_number)
        )
    
    async def fuzzy_match_brand(self, text: str) -> Tuple[Optional[str], float]:
        best_match = None
        best_score = 0
        text_lower = text.lower()
        text_words = text_lower.split()
        logger.debug(f"Attempting to fuzzy match: '{text}'")

        def get_consonants(s):
            return ''.join(c for c in s if c.isalpha() and c.lower() not in 'aeiou')

        def word_match_score(text_words, brand_words):
            scores = []
            for brand_word in brand_words:
                best_word_score = max(fuzz.ratio(brand_word, text_word) for text_word in text_words)
                scores.append(best_word_score)
            return sum(scores) / len(scores) if scores else 0

        for brand, brand_info in self.brand_database.items():
            # Check for excluded words
            if any(excluded_word.lower() in text_lower for excluded_word in brand_info.get('word_exclusions', [])):
                logger.debug(f"Skipping '{brand}' due to excluded word in text")
                continue

            # Combine brand name with its variations, avoiding duplication
            brand_lower = brand.lower()
            variations = [var.lower() for var in brand_info.get('variations', [])]
            brand_variations = [brand_lower] + [var for var in variations if var != brand_lower]

            for brand_variation in brand_variations:
                brand_words = brand_variation.split()
                is_multi_word = len(brand_words) > 1

                # Full string comparison
                ratio = fuzz.ratio(text_lower, brand_variation)
                
                # Word-by-word matching for multi-word brands
                word_score = word_match_score(text_words, brand_words) if is_multi_word else 0
                
                # Calculate initial score
                if is_multi_word:
                    score = (ratio * 0.3 + word_score * 0.7)  # Favor word-by-word matching for multi-word brands
                else:
                    score = ratio
                
                # Exact word match bonus (increased and applied to all brands)
                exact_matches = sum(word in text_words for word in brand_words)
                if exact_matches > 0:
                    exact_match_bonus = 40 * (exact_matches / len(brand_words))
                    score += exact_match_bonus
                    logger.debug(f"Exact word match bonus applied for '{brand_variation}'. Bonus: {exact_match_bonus}")
                
                # Additional bonus for single-word brands that match exactly
                if not is_multi_word and brand_variation in text_words:
                    score += 30
                    logger.debug(f"Single-word exact match bonus applied for '{brand_variation}'")
                
                # Length difference penalty (less aggressive for multi-word brands)
                len_diff = abs(len(text_lower) - len(brand_variation))
                length_penalty = min(20 if is_multi_word else 30, (len_diff / max(len(text_lower), len(brand_variation))) * 100)
                score = max(0, score - length_penalty)
                
                # Consonant matching
                text_consonants = get_consonants(text_lower)
                brand_consonants = get_consonants(brand_variation)
                consonant_ratio = fuzz.ratio(text_consonants, brand_consonants)
                score = (score * 0.8 + consonant_ratio * 0.2)
                
                # Multi-word brand bonus
                if is_multi_word and any(word in text_lower for word in brand_words):
                    score += 15
                    logger.debug(f"Multi-word brand bonus applied for '{brand_variation}'. New score: {score}")
                
                logger.debug(f"Fuzzy match for '{brand}' (variation: '{brand_variation}'). Score: {score}")

                # Penalty for very short matches (only for single-word brands)
                if not is_multi_word and (len(text_lower) <= 3 or len(brand_variation) <= 3):
                    score *= 0.7  # Reduced penalty
                    logger.debug(f"Penalty applied for short string. New score: {score}")

                if score > best_score:
                    best_score = score
                    best_match = brand  # Store the main brand name, not the variation
                    logger.debug(f"New best match: '{best_match}' with score {best_score}")

        # Apply a minimum threshold
        if best_score < settings.MINIMUM_FUZZY_BRAND_MATCH_SCORE:
            logger.debug(f"No fuzzy match found for '{text}' (best score: {best_score})")
            return None, 0

        logger.debug(f"Best fuzzy match: '{text}' -> '{best_match}', Score: {best_score}")
        return best_match, best_score
    
    async def remove_over_interpolated_brands(self, brands: List[BrandInstance]) -> List[BrandInstance]:
        filtered_brands = []
        current_sequence = []
        for brand in brands:
            if brand.is_interpolated:
                current_sequence.append(brand)
            else:
                if current_sequence:
                    sequence_length = brand.frame_number - current_sequence[0].frame_number
                    if sequence_length <= settings.INTERPOLATION_LIMIT:
                        filtered_brands.extend(current_sequence)
                    current_sequence = []
                filtered_brands.append(brand)
        
        if current_sequence:
            sequence_length = brands[-1].frame_number - current_sequence[0].frame_number
            if sequence_length <= settings.INTERPOLATION_LIMIT:
                filtered_brands.extend(current_sequence)

        return filtered_brands
    
    async def calculate_relative_size(self, bounding_box: Dict) -> float:
        box_width = bounding_box['vertices'][1]['x'] - bounding_box['vertices'][0]['x']
        box_height = bounding_box['vertices'][2]['y'] - bounding_box['vertices'][0]['y']
        return (box_width * box_height) / (self.frame_width * self.frame_height)

    async def calculate_position(self, bounding_box: Dict) -> Dict[str, float]:
        center_x = (bounding_box['vertices'][0]['x'] + bounding_box['vertices'][1]['x']) / 2
        center_y = (bounding_box['vertices'][0]['y'] + bounding_box['vertices'][2]['y']) / 2
        return {
            "x": center_x / self.frame_width,
            "y": center_y / self.frame_height
        }

    async def convert_relative_bbox(self, bbox: Dict) -> Dict:
        """
        Convert relative bounding box coordinates to absolute pixel values.
        """
        try:
            if self.video_resolution is None:
                raise ValueError("Video resolution is not set")
            
            video_width, video_height = self.video_resolution
            left = bbox.get('Left', 0) * video_width
            top = bbox.get('Top', 0) * video_height
            width = bbox.get('Width', 0) * video_width
            height = bbox.get('Height', 0) * video_height
            vertices = [
                {"x": int(left), "y": int(top)},
                {"x": int(left + width), "y": int(top)},
                {"x": int(left + width), "y": int(top + height)},
                {"x": int(left), "y": int(top + height)}
            ]
            logger.debug(f"Converted bbox {bbox} to vertices {vertices} for resolution {self.video_resolution}")
            return {"vertices": vertices}
        except Exception as e:
            logger.error(f"Error converting bounding box {bbox} for resolution {self.video_resolution}: {str(e)}", exc_info=True)
            # Return an empty bounding box in case of error
            return {"vertices": []}
        
    def get_all_brand_detections(self):
        return self.all_brand_detections

    def get_accumulated_results(self):
        return self.accumulated_results

    def add_brand_detection(self, brand: BrandInstance):
        self.all_brand_detections.append(brand)

    def add_to_accumulated_results(self, frame: int, brand: BrandInstance):
        if frame not in self.accumulated_results:
            self.accumulated_results[frame] = []
        self.accumulated_results[frame].append(brand)

    def remove_brand_from_accumulated_results(self, frame: int, brand: str):
        if frame in self.accumulated_results:
            self.accumulated_results[frame] = [b for b in self.accumulated_results[frame] if b.brand != brand]

    def sort_all_brand_detections(self):
        self.all_brand_detections.sort(key=lambda x: x.frame_number)

    @staticmethod
    async def clean_text(text: str) -> str:
        if text is None:
            return ''
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()
        if len(cleaned_text) < 3:
            logger.debug(f"Text too short for brand matching: '{text}'")
            return ''
        return cleaned_text

    @staticmethod
    async def calculate_box_overlap(box1: Dict, box2: Dict) -> float:
        try:
            logger.debug("Calculating overlap between two bounding boxes")

            def calculate_overlap_1d(min1, max1, min2, max2):
                return max(0, min(max1, max2) - max(min1, min2))

            # Calculate x-axis overlap
            x_overlap = calculate_overlap_1d(
                min(box1['vertices'][0]['x'], box1['vertices'][3]['x']),
                max(box1['vertices'][1]['x'], box1['vertices'][2]['x']),
                min(box2['vertices'][0]['x'], box2['vertices'][3]['x']),
                max(box2['vertices'][1]['x'], box2['vertices'][2]['x'])
            )

            # Calculate y-axis overlap
            y_overlap = calculate_overlap_1d(
                min(box1['vertices'][0]['y'], box1['vertices'][1]['y']),
                max(box1['vertices'][2]['y'], box1['vertices'][3]['y']),
                min(box2['vertices'][0]['y'], box2['vertices'][1]['y']),
                max(box2['vertices'][2]['y'], box2['vertices'][3]['y'])
            )

            # Calculate overlap area
            overlap_area = x_overlap * y_overlap
            logger.debug(f"Calculated overlap area: {overlap_area}")

            return overlap_area

        except KeyError as e:
            logger.error(f"BrandDetector: KeyError in calculate_box_overlap: {str(e)}. Ensure both boxes have 'vertices' with correct structure.", exc_info=True)
            return 0.0  # Return no overlap in case of error
        except Exception as e:
            logger.error(f"BrandDetector: Error in calculate_box_overlap: {str(e)}", exc_info=True)
            return 0.0  # Return no overlap in case of error
    
    @staticmethod
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

            logger.debug(f"Box1 text: '{box1_text}', Box2 text: '{box2_text}'")
            logger.debug(f"Box1 full text: '{box1_full_text}', Box2 full text: '{box2_full_text}'")

            # Check if cleaned_text matches the full brand name in the text field
            if box1_text == box1_full_text and box2_text == box2_full_text:
                logger.debug("Boxes should not be merged as they contain the full brand name already")
                return False

            # Check if the combined cleaned_text of both boxes matches the text field of either box
            combined_text = f"{box1_text} {box2_text}".strip()
            if combined_text == box1_full_text or combined_text == box2_full_text:
                logger.debug("Combined cleaned text matches full brand name, proceeding to evaluate proximity and overlap")
            else:
                logger.debug(f"Combined text '{combined_text}' does not match either full text, but proceeding with proximity check")

            # Check for overlap
            overlap_area = await BrandDetector.calculate_box_overlap(box1, box2)
            box1_area = BrandDetector.calculate_box_area(v1)
            box2_area = BrandDetector.calculate_box_area(v2)
            min_area = min(box1_area, box2_area)

            logger.debug(f"Overlap area: {overlap_area}, Box1 area: {box1_area}, Box2 area: {box2_area}")

            if overlap_area > 0 and overlap_area / min_area >= settings.MIN_OVERLAP_RATIO_FOR_MERGE:
                logger.debug("Boxes should be merged due to significant overlap")
                return True

            # Check if bottom of box1 is close to top of box2 or vice versa
            vertical_distance = min(abs(max(v1[2]['y'], v1[3]['y']) - min(v2[0]['y'], v2[1]['y'])),
                                    abs(max(v2[2]['y'], v2[3]['y']) - min(v1[0]['y'], v1[1]['y'])))
            if close_edges(vertical_distance, 0, frame_height):
                logger.debug(f"Boxes should be merged due to close vertical edges. Distance: {vertical_distance}")
                return True

            # Check if right of box1 is close to left of box2 or vice versa
            horizontal_distance = min(abs(max(v1[1]['x'], v1[2]['x']) - min(v2[0]['x'], v2[3]['x'])),
                                    abs(max(v2[1]['x'], v2[2]['x']) - min(v1[0]['x'], v1[3]['x'])))
            if close_edges(horizontal_distance, 0, frame_width):
                logger.debug(f"Boxes should be merged due to close horizontal edges. Distance: {horizontal_distance}")
                return True

            logger.debug("Boxes should not be merged")
            return False

        except Exception as e:
            logger.error(f"Error in should_merge_bounding_boxes: {str(e)}", exc_info=True)
            return False  # Return False in case of error
        
    @staticmethod
    async def merge_bounding_boxes(box1: Dict, box2: Dict) -> Tuple[Dict, List[Dict]]:
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

            merged_from = [box1, box2]

            logger.debug(f"Merged bounding box: {merged_box}")
            return merged_box, merged_from

        except Exception as e:
            logger.error(f"Error in merge_bounding_boxes: {str(e)}", exc_info=True)
            return box1, [box1]  # Return first box in case of error

    @staticmethod
    def calculate_box_area(vertices):
        width = max(v['x'] for v in vertices) - min(v['x'] for v in vertices)
        height = max(v['y'] for v in vertices) - min(v['y'] for v in vertices)
        return width * height
########################################################