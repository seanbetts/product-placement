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

        # Enable logging for a specific brand and frames
        # DO_LOGGING = False
        # LOGGING_BRAND: Optional[str] = 'doritos'
        # LOGGING_START_FRAME = 663
        # LOGGING_END_FRAME = 663

        # Default logging to false to avoid logging for every frame
        DO_LOGGING = False

        # Try to get the logging frame range, use default values if not set
        try:
            LOGGING_START_FRAME = LOGGING_START_FRAME  # This will raise a NameError if not defined
            LOGGING_END_FRAME = LOGGING_END_FRAME  # This will raise a NameError if not defined
            frame_logging_enabled = True
        except NameError:
            LOGGING_START_FRAME = 0
            LOGGING_END_FRAME = -1  # Use -1 as a sentinel value
            frame_logging_enabled = False
            DO_LOGGING = False

        # Log the frame number if frame logging is enabled and the frame is within the specified range
        if frame_logging_enabled and LOGGING_START_FRAME <= frame.number <= LOGGING_END_FRAME:
            logger.info(f"--- Processing frame {frame.number} ---")
            DO_LOGGING = True
        else:
            DO_LOGGING = False

        text_detections = frame.ocr_data['rekognition_response'].get('TextDetections', [])
        logger.debug(f"Frame {frame.number}: Number of text detections: {len(text_detections)}")

        line_detections = [det for det in text_detections if det['Type'] == 'LINE']
        cleaned_detections = [await self.clean_detection(det) for det in line_detections]
        cleaned_detections = [det for det in cleaned_detections if det is not None]
        logger.debug(f"Frame {frame.number}: Cleaned detections: {len(cleaned_detections)} items")

        combined_detections = await self.combine_adjacent_detections(cleaned_detections)
        logger.debug(f"Frame {frame.number}: Combined detections: {len(combined_detections)} items")

        all_detections = cleaned_detections + combined_detections
        matched_brands = []
        for det in all_detections:
            if 'LOGGING_BRAND' in locals() or 'LOGGING_BRAND' in globals():
                brand = await self.match_brand(det, logging_brand=LOGGING_BRAND, logging_start_frame=LOGGING_START_FRAME, logging_end_frame=LOGGING_END_FRAME, do_logging=DO_LOGGING)
            else:
                brand = await self.match_brand(det, logging_start_frame=LOGGING_START_FRAME, logging_end_frame=LOGGING_END_FRAME, do_logging=DO_LOGGING)
            if brand:
                matched_brands.append(brand)

        if DO_LOGGING:
            logger.info(f"Video Processing - Brand Detection - Step 3.2: Frame {frame.number} - Matched brands before grouping: {[b.cleaned_text for b in matched_brands]}")

        # Group overlapping detections
        grouped_detections = await self.group_overlapping_detections(matched_brands)

        logger.debug(f"Frame {frame.number}: Grouped detections: {[len(group) for group in grouped_detections]}")

        # Remove duplicates within each group
        filtered_brands = []
        all_discarded_brands = []
        for group in grouped_detections:
            filtered_group, discarded_group = await self.remove_duplicate_detections(group)
            filtered_brands.extend(filtered_group)
            all_discarded_brands.extend(discarded_group)

        # Additional step to remove any remaining duplicates
        final_filtered_brands = []
        for brand in filtered_brands:
            is_duplicate = False
            for existing in final_filtered_brands:
                if await self.is_duplicate(brand, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_filtered_brands.append(brand)

        if DO_LOGGING:
            logger.info(f"Video Processing - Brand Detection - Step 3.2: Frame {frame.number} - Final filtered brands: {[b.cleaned_text for b in final_filtered_brands]}")

        # Use final_filtered_brands instead of filtered_brands in subsequent processing
        self.all_detections[frame.number] = final_filtered_brands

        # Group detections by brand
        brand_groups = {}
        for brand in filtered_brands:
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

        if DO_LOGGING:
            logger.info(f"Video Processing - Brand Detection - Step 3.2: Frame {frame.number} - Merged brands: {[b.cleaned_text for b in merged_brands]}")

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

        # Accumulate brand appearances
        for brand in interpolated_brands:
            self.brand_appearances[brand.brand].add(frame.number)
        self.all_brand_detections.extend(interpolated_brands)

        self.accumulated_results[frame.number] = interpolated_brands

        if DO_LOGGING:
            brand_results = [
                f"Brand: {brand.brand}, Confidence: {brand.brand_match_confidence:.2f}"
                for brand in self.accumulated_results[frame.number]
            ]
            brand_results_str = ", ".join(brand_results)

            logger.info(f"Video Processing - Brand Detection - Step 3.2: Frame {frame.number} - Brands: {brand_results_str}")

        return DetectionResult(frame.number, self.accumulated_results[frame.number])

    async def clean_detection(self, detection: Dict) -> Optional[Dict]:
        try:
            logger.debug(f"Cleaning detection: {detection['Type']} - {detection['DetectedText']}")
            
            if detection['Type'] != settings.OCR_TYPE:
                return None

            confidence = detection['Confidence']
            
            if confidence < settings.MINIMUM_OCR_CONFIDENCE:
                logger.debug(f"Skipping detection due to low confidence: {confidence}")
                return None

            original_text = detection['DetectedText'].lower()

            if confidence > settings.MAX_CLEANING_CONFIDENCE:
                cleaned_text = original_text
            else:
                cleaned_text = await self.clean_text(original_text)

            logger.debug(f"Frame {self.current_frame_number}: Cleaned text: '{original_text}' -> '{cleaned_text}'")

            if cleaned_text is None or len(cleaned_text.strip()) <= 2:
                logger.debug(f"Skipping detection due to short or None cleaned text: '{original_text}'")
                return None

            bounding_box = await self.convert_relative_bbox(detection['Geometry']['BoundingBox'])
            adapted_bounding_box = self.adapt_bounding_box(bounding_box)

            return {
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "confidence": confidence,
                "bounding_box": adapted_bounding_box,
                "frame_number": self.current_frame_number
            }
        except Exception as e:
            logger.error(f"Error in clean_detection: {str(e)}", exc_info=True)
            return None
        
    async def combine_adjacent_detections(self, detections: List[Dict]) -> List[Dict]:
        combined = []
        sorted_detections = sorted(detections, key=lambda x: x['bounding_box']['vertices'][0]['y'])
        i = 0
        while i < len(sorted_detections):
            current = sorted_detections[i]
            combined_text = current['cleaned_text']
            combined_confidence = current['confidence']
            combined_box = current['bounding_box'].copy()
            
            # Check if the current detection is already a complete brand name
            if any(combined_text.lower() == brand.lower() for brand in self.brand_database.keys()):
                combined.append({
                    'cleaned_text': combined_text,
                    'confidence': combined_confidence,
                    'bounding_box': combined_box,
                    'frame_number': self.current_frame_number
                })
                i += 1
                continue

            j = i + 1
            while j < len(sorted_detections):
                next_det = sorted_detections[j]
                if await self.should_combine_detections(current, next_det):
                    combined_text += ' ' + next_det['cleaned_text']
                    combined_confidence = (combined_confidence + next_det['confidence']) / 2
                    try:
                        combined_box, _ = await BrandDetector.merge_bounding_boxes(combined_box, next_det['bounding_box'])
                    except Exception as e:
                        logger.error(f"Error merging bounding boxes: {str(e)}")
                        break
                    
                    # If we've formed a complete brand name, stop combining
                    if any(combined_text.lower() == brand.lower() for brand in self.brand_database.keys()):
                        break
                    
                    j += 1
                else:
                    break
            
            combined.append({
                'cleaned_text': combined_text,
                'confidence': combined_confidence,
                'bounding_box': combined_box,
                'frame_number': self.current_frame_number
            })
            i = j
        
        return combined
    
    async def should_combine_detections(self, det1: Dict, det2: Dict) -> bool:
        # Check if the detections are adjacent
        if not await self.are_detections_adjacent(det1, det2):
            return False

        # Get the cleaned text for both detections
        text1 = det1['cleaned_text'].lower()
        text2 = det2['cleaned_text'].lower()

        # Combine the texts
        combined_text = f"{text1} {text2}"

        # Check if the combination exactly matches a brand name
        return any(combined_text == brand.lower() for brand in self.brand_database.keys())
        
    async def are_detections_adjacent(self, det1: Dict, det2: Dict) -> bool:
        v1 = det1['bounding_box'].get('vertices', [])
        v2 = det2['bounding_box'].get('vertices', [])

        if isinstance(v1[0], dict):
            y1_bottom = max(v['y'] for v in v1)
            y2_top = min(v['y'] for v in v2)
            x1_left = min(v['x'] for v in v1)
            x1_right = max(v['x'] for v in v1)
            x2_left = min(v['x'] for v in v2)
            x2_right = max(v['x'] for v in v2)
        else:
            y1_bottom = max(v[1] for v in v1)
            y2_top = min(v[1] for v in v2)
            x1_left = min(v[0] for v in v1)
            x1_right = max(v[0] for v in v1)
            x2_left = min(v[0] for v in v2)
            x2_right = max(v[0] for v in v2)

        # Check vertical adjacency
        vertical_distance = y2_top - y1_bottom
        max_vertical_distance = settings.MIN_VERTICAL_OVERLAP_RATIO_FOR_MERGE * self.video_resolution[1]
        is_vertically_adjacent = vertical_distance <= max_vertical_distance

        # Check horizontal overlap
        overlap_left = max(x1_left, x2_left)
        overlap_right = min(x1_right, x2_right)
        overlap_width = max(0, overlap_right - overlap_left)
        min_width = min(x1_right - x1_left, x2_right - x2_left)
        horizontal_overlap_ratio = overlap_width / min_width if min_width > 0 else 0
        is_horizontally_overlapping = horizontal_overlap_ratio >= settings.MIN_HORIZONTAL_OVERLAP_RATIO_FOR_MERGE

        return is_vertically_adjacent and is_horizontally_overlapping

    async def match_brand(self, detection: Dict, logging_brand: Optional[str] = None, logging_start_frame: Optional[int] = None, logging_end_frame: Optional[int] = None, do_logging: Optional[bool] = None) -> Optional[BrandInstance]:
        text = detection['cleaned_text']
        brand_match, brand_score = await self.fuzzy_match_brand(text, logging_brand, do_logging)

        if brand_match == logging_brand and logging_start_frame <= self.current_frame_number <= logging_end_frame:
            logger.debug(f"Frame {self.current_frame_number}: Matched brand: {brand_match} with Brand score: {brand_score}")
        
        if brand_match and brand_score >= settings.MINIMUM_FUZZY_BRAND_MATCH_SCORE:
            combined_confidence = (brand_score + detection['confidence']) / 2
            
            logger.debug(f"Frame {self.current_frame_number}: Matched brand: {brand_match} with combined confidence {combined_confidence}")
            relative_size = await self.calculate_relative_size(detection['bounding_box'])
            position = await self.calculate_position(detection['bounding_box'])
            frame_timestamp = self.current_frame_number / self.fps
            
            return BrandInstance(
                brand=brand_match,
                brand_match_confidence=combined_confidence,
                original_detected_text=detection['cleaned_text'],
                original_detected_text_confidence=detection['confidence'],
                cleaned_text=detection['cleaned_text'],
                bounding_box=detection['bounding_box'],
                frame_number=self.current_frame_number,
                detection_type="OCR",
                relative_size=relative_size,
                position=position,
                frame_timestamp=frame_timestamp,
                fuzzy_match_term=text,
                fuzzy_match_score=brand_score,
                is_interpolated=False
            )
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
    
    async def fuzzy_match_brand(self, text: str, logging_brand: Optional[str] = None, do_logging: Optional[bool] = None) -> Tuple[Optional[str], float]:
        MIN_TEXT_LENGTH = settings.MIN_TEXT_LENGTH
        MIN_BRAND_LENGTH = settings.MIN_BRAND_LENGTH
        FULL_MATCH_WEIGHT = settings.FULL_MATCH_WEIGHT
        PARTIAL_MATCH_WEIGHT = settings.PARTIAL_MATCH_WEIGHT
        WORD_MATCH_WEIGHT = settings.WORD_MATCH_WEIGHT
        MIN_FULL_MATCH_SCORE = settings.MIN_FULL_MATCH_SCORE
        MIN_PARTIAL_MATCH_SCORE = settings.MIN_PARTIAL_MATCH_SCORE
        MIN_WORD_MATCH_SCORE = settings.MIN_WORD_MATCH_SCORE
        LENGTH_PENALTY_FACTOR = settings.LENGTH_PENALTY_FACTOR
        EXACT_MATCH_BONUS = settings.EXACT_MATCH_BONUS
        CONTAINS_BRAND_BONUS = settings.CONTAINS_BRAND_BONUS
        WORD_IN_BRAND_BONUS = settings.WORD_IN_BRAND_BONUS
        COMMON_WORDS = settings.COMMON_WORDS

        best_match = None
        best_score = 0
        text_lower = text.lower()
        text_words = [word for word in text_lower.split() if word not in COMMON_WORDS]
        
        if len(text_lower) < MIN_TEXT_LENGTH:
            logger.debug(f"Text too short for brand matching: '{text}'")
            return None, 0

        if do_logging:
            logger.debug(f"Attempting to fuzzy match: '{text}'")

        for brand, brand_info in self.brand_database.items():
            brand_lower = brand.lower()
            brand_words = brand_lower.split()
            variations = brand_info.get('variations', [])
            
            if len(brand_lower) < MIN_BRAND_LENGTH:
                continue

            # Full brand match (including variations)
            full_match_scores = [fuzz.ratio(text_lower, brand_lower)] + [fuzz.ratio(text_lower, var.lower()) for var in variations]
            full_match_score = max(full_match_scores)

            # Partial brand match (including variations)
            partial_scores = [fuzz.partial_ratio(text_lower, brand_lower)] + [fuzz.partial_ratio(text_lower, var.lower()) for var in variations]
            for i in range(len(text_words)):
                for j in range(i+1, len(text_words)+1):
                    partial_text = ' '.join(text_words[i:j])
                    partial_scores.extend([fuzz.partial_ratio(partial_text, brand_lower)] + [fuzz.partial_ratio(partial_text, var.lower()) for var in variations])
            
            partial_match_score = max(partial_scores) if partial_scores else 0
            
            # Word-by-word match (including variations)
            brand_words = [word for word in brand_lower.split() if word not in COMMON_WORDS]
            variation_words = [word for var in variations for word in var.lower().split() if word not in COMMON_WORDS]
            all_brand_words = set(brand_words + variation_words)
            word_scores = [max(fuzz.ratio(text_word, brand_word) for brand_word in all_brand_words) for text_word in text_words]
            avg_word_score = sum(word_scores) / len(word_scores) if word_scores else 0
            
            # Apply length penalty
            length_penalty = LENGTH_PENALTY_FACTOR * (1 / len(brand_lower) + 1 / len(text_lower))
            
            # Add a score for matching individual words in multi-word brands
            brand_word_match_score = 0
            if len(brand_words) > 1:
                for word in text_lower.split():
                    if word in brand_words:
                        brand_word_match_score += 100  # Full score for matching a word in the brand name

            # Adjust the weighted score calculation
            weighted_score = (
                FULL_MATCH_WEIGHT * full_match_score +
                PARTIAL_MATCH_WEIGHT * partial_match_score +
                WORD_MATCH_WEIGHT * avg_word_score +
                (brand_word_match_score / len(brand_words))  # Average score for matching words
            ) * (1 - length_penalty)
            
            # Apply minimum score thresholds
            if (full_match_score < MIN_FULL_MATCH_SCORE or
                partial_match_score < MIN_PARTIAL_MATCH_SCORE or
                avg_word_score < MIN_WORD_MATCH_SCORE):
                weighted_score *= 0.5

            # Consider the ratio of matched length to total length
            match_ratio = len(brand_lower) / len(text_lower)
            if match_ratio < 0.5 or match_ratio > 2:
                weighted_score *= 0.75

            # Apply bonus for exact matches
            exact_match = text_lower == brand_lower or text_lower in [var.lower() for var in variations]
            if exact_match:
                weighted_score += EXACT_MATCH_BONUS

            # Apply bonus for containing the brand name
            contains_brand = brand_lower in text_lower or any(var.lower() in text_lower for var in variations)
            if contains_brand:
                weighted_score += CONTAINS_BRAND_BONUS

            # Apply bonus for word-in-brand match
            if any(word in brand_words for word in text_words):
                weighted_score += WORD_IN_BRAND_BONUS
                    
            if do_logging:
                logger.debug(f"Brand: {brand}, Full Score: {full_match_score}, Partial Score: {partial_match_score}, "
                        f"Avg Word Score: {avg_word_score}, Weighted Score: {weighted_score}")
            
            is_current_best_match = weighted_score > best_score
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_match = brand

            # Detailed logging for specified brand
            if do_logging and logging_brand and brand.lower() == logging_brand.lower() and (text_lower == brand_lower or text_lower in [var.lower() for var in variations]):
                logger.info(f"Fuzzy match found for '{text}' (Best score: {best_score:.2f})")
                logger.info(f"--- Detailed Logging for {brand.capitalize()} ---")
                logger.info(f"Input text: '{text}'")
                logger.info(f"Variations: {variations}")
                logger.info(f"Full match scores: {full_match_scores}")
                logger.info(f"Full match score (max): {full_match_score}")
                logger.info(f"Partial match score: {partial_match_score}")
                logger.info(f"Word scores: {word_scores}")
                logger.info(f"Average word score: {avg_word_score}")
                logger.info(f"Length penalty: {length_penalty:.2f}")
                logger.info(f"Match ratio: {match_ratio}")
                logger.info(f"Exact match: {exact_match}")
                logger.info(f"Contains brand: {contains_brand}")
                logger.info(f"Weighted score before bonuses: {(weighted_score - (EXACT_MATCH_BONUS if exact_match else 0) - (CONTAINS_BRAND_BONUS if contains_brand else 0)):.2f}")
                logger.info(f"Final weighted score: {weighted_score:.2f}")
                logger.info(f"Current best score: {best_score:.2f}")
                logger.info(f"Is current best match: {is_current_best_match}")
                logger.info("-----------------------------------")

        if do_logging and best_score < settings.MINIMUM_FUZZY_BRAND_MATCH_SCORE:
            logger.info(f"No fuzzy match found for '{text}' (Best score: {best_score:.2f})")
            return None, 0

        logger.debug(f"Best fuzzy match: '{text}' -> '{best_match}', Score: {best_score}")
        return best_match, best_score
    
    async def calculate_relative_size(self, bounding_box: Dict) -> float:
        vertices = bounding_box.get('vertices', [])
        if not vertices:
            logger.error(f"Invalid bounding box format: {bounding_box}")
            return 0.0
        
        if isinstance(vertices[0], dict):
            box_width = vertices[1]['x'] - vertices[0]['x']
            box_height = vertices[2]['y'] - vertices[0]['y']
        else:
            box_width = vertices[1][0] - vertices[0][0]
            box_height = vertices[2][1] - vertices[0][1]
        
        return (box_width * box_height) / (self.frame_width * self.frame_height)
    
    async def group_overlapping_detections(self, detections: List[BrandInstance]) -> List[List[BrandInstance]]:
        groups = []
        for detection in detections:
            added_to_group = False
            for group in groups:
                overlap_found = False
                for other in group:
                    overlap = await self.calculate_box_overlap(detection.bounding_box, other.bounding_box)
                    if overlap > 0:
                        overlap_found = True
                        break
                if overlap_found:
                    group.append(detection)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([detection])
        return groups
    
    async def remove_duplicate_detections(self, group: List[BrandInstance]) -> Tuple[List[BrandInstance], List[BrandInstance]]:
        logger.debug(f"Starting remove_duplicate_detections with {len(group)} detections")
        if len(group) <= 1:
            logger.debug("Only one or zero detections, returning group as-is")
            return group, []

        def detection_score(detection):
            area = self.calculate_box_area(self.adapt_bounding_box(detection.bounding_box))
            score = (detection.brand_match_confidence, len(detection.cleaned_text.split()), area)
            logger.debug(f"Detection score for '{detection.cleaned_text}': {score}")
            return score

        sorted_group = sorted(group, key=detection_score, reverse=True)
        logger.debug(f"Sorted group: {[d.cleaned_text for d in sorted_group]}")
        
        filtered_detections = []
        discarded_detections = []

        for i, detection in enumerate(sorted_group):
            logger.debug(f"Evaluating detection {i}: '{detection.cleaned_text}'")
            should_keep = True
            for kept_detection in filtered_detections:
                overlap_ratio = await self.calculate_overlap_ratio(detection.bounding_box, kept_detection.bounding_box)
                text_similarity = fuzz.ratio(detection.cleaned_text.lower(), kept_detection.cleaned_text.lower()) / 100.0
                logger.debug(f"Overlap ratio: {overlap_ratio}, Text similarity: {text_similarity}")

                if overlap_ratio > 0.5 or text_similarity > 0.8:
                    logger.debug(f"Significant overlap or text similarity found")
                    if len(detection.cleaned_text) <= len(kept_detection.cleaned_text):
                        logger.debug(f"Discarding '{detection.cleaned_text}' as it's a subset or duplicate of '{kept_detection.cleaned_text}'")
                        should_keep = False
                        break
                    else:
                        logger.debug(f"Replacing '{kept_detection.cleaned_text}' with '{detection.cleaned_text}'")
                        filtered_detections.remove(kept_detection)
                        discarded_detections.append(kept_detection)
                        break

            if should_keep:
                logger.debug(f"Keeping detection: '{detection.cleaned_text}'")
                filtered_detections.append(detection)
            else:
                logger.debug(f"Discarding detection: '{detection.cleaned_text}'")
                discarded_detections.append(detection)

        logger.debug(f"Final filtered detections: {[d.cleaned_text for d in filtered_detections]}")
        logger.debug(f"Discarded detections: {[d.cleaned_text for d in discarded_detections]}")
        return filtered_detections, discarded_detections

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
        
    async def calculate_overlap_ratio(self, box1: Dict, box2: Dict) -> float:
        box1 = self.adapt_bounding_box(box1)
        box2 = self.adapt_bounding_box(box2)
        
        overlap_area = await self.calculate_box_overlap(box1, box2)
        box1_area = self.calculate_box_area(box1)
        box2_area = self.calculate_box_area(box2)
        
        logger.debug(f"Overlap area: {overlap_area}, Box1 area: {box1_area}, Box2 area: {box2_area}")
        
        if box1_area == 0 or box2_area == 0:
            logger.debug("One of the boxes has zero area, returning overlap ratio 0")
            return 0
        
        ratio = overlap_area / min(box1_area, box2_area)
        logger.debug(f"Calculated overlap ratio: {ratio}")
        return ratio
    
    async def is_duplicate(self, brand1: BrandInstance, brand2: BrandInstance) -> bool:
        text_similarity = fuzz.ratio(brand1.cleaned_text.lower(), brand2.cleaned_text.lower()) / 100.0
        overlap_ratio = await self.calculate_overlap_ratio(brand1.bounding_box, brand2.bounding_box)
        return (text_similarity > 0.8 and overlap_ratio > 0.5) or brand1.cleaned_text in brand2.cleaned_text or brand2.cleaned_text in brand1.cleaned_text
        
    def get_all_brand_detections(self):
        return self.all_brand_detections
    
    def set_accumulated_results(self, results: Dict[int, List[BrandInstance]]):
        self.accumulated_results = results

    def get_accumulated_results(self) -> Dict[int, List[BrandInstance]]:
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

    def adapt_bounding_box(self, bounding_box: Dict) -> Dict:
        """
        Adapt the bounding box format to ensure compatibility with existing functions.
        """
        if 'vertices' in bounding_box and isinstance(bounding_box['vertices'], list):
            return bounding_box  # Already in the correct format
        
        # Convert to the format expected by other functions
        vertices = bounding_box.get('vertices', [])
        if isinstance(vertices, tuple):
            vertices = list(vertices)
        
        return {
            'vertices': vertices
        }
    
    def is_box_contained(self, box1: Dict, box2: Dict) -> bool:
        v1 = box1.get('vertices', [])
        v2 = box2.get('vertices', [])
        
        if len(v1) < 4 or len(v2) < 4:
            return False  # Cannot determine containment with insufficient vertices
        
        return (v1[0]['x'] >= v2[0]['x'] and
                v1[0]['y'] >= v2[0]['y'] and
                v1[2]['x'] <= v2[2]['x'] and
                v1[2]['y'] <= v2[2]['y'])
    
    def find_previous_non_interpolated_frame(self, brand: str, current_frame: int) -> Optional[int]:
        for frame in range(current_frame - 1, -1, -1):
            brands = self.get_accumulated_results().get(frame, [])
            for b in brands:
                if b.brand == brand and not b.is_interpolated:
                    return frame
        return None

    def find_next_non_interpolated_frame(self, brand: str, current_frame: int) -> Optional[int]:
        max_frame = max(self.get_accumulated_results().keys())
        for frame in range(current_frame + 1, max_frame + 1):
            brands = self.get_accumulated_results().get(frame, [])
            for b in brands:
                if b.brand == brand and not b.is_interpolated:
                    return frame
        return None

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

            # Check if one box is fully contained within the other
            if BrandDetector.is_box_contained(box1, box2) or BrandDetector.is_box_contained(box2, box1):
                logger.debug("One box is fully contained within the other, should not merge")
                return False

            if overlap_area > 0 and overlap_area / min_area >= settings.MIN_VERTICAL_OVERLAP_RATIO_FOR_MERGE:
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

            v1, v2 = box1.get('vertices', []), box2.get('vertices', [])
            
            if not v1 or not v2:
                raise ValueError("Invalid bounding box format")

            if isinstance(v1[0], dict):
                merged_box = {
                    'vertices': [
                        {'x': min(v1[0]['x'], v2[0]['x']), 'y': min(v1[0]['y'], v2[0]['y'])},
                        {'x': max(v1[1]['x'], v2[1]['x']), 'y': min(v1[1]['y'], v2[1]['y'])},
                        {'x': max(v1[2]['x'], v2[2]['x']), 'y': max(v1[2]['y'], v2[2]['y'])},
                        {'x': min(v1[3]['x'], v2[3]['x']), 'y': max(v1[3]['y'], v2[3]['y'])}
                    ]
                }
            else:
                merged_box = {
                    'vertices': [
                        (min(v1[0][0], v2[0][0]), min(v1[0][1], v2[0][1])),
                        (max(v1[1][0], v2[1][0]), min(v1[1][1], v2[1][1])),
                        (max(v1[2][0], v2[2][0]), max(v1[2][1], v2[2][1])),
                        (min(v1[3][0], v2[3][0]), max(v1[3][1], v2[3][1]))
                    ]
                }

            merged_from = [box1, box2]

            logger.debug(f"Merged bounding box: {merged_box}")
            return merged_box, merged_from

        except Exception as e:
            logger.error(f"Error in merge_bounding_boxes: {str(e)}", exc_info=True)
            return box1, [box1]  # Return first box in case of error

    @staticmethod
    def calculate_box_area(bounding_box: Dict) -> float:
        vertices = bounding_box.get('vertices', [])
        if len(vertices) < 4:
            return 0  # Return 0 area if there are not enough vertices
        
        width = max(v['x'] for v in vertices) - min(v['x'] for v in vertices)
        height = max(v['y'] for v in vertices) - min(v['y'] for v in vertices)
        return width * height
########################################################