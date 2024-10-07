import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
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
    text: str
    confidence: float
    bounding_box: Dict
    original_text: str
    cleaned_text: str
    frame_number: int
    is_interpolated: bool = False

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "frame_number": self.frame_number,
            "is_interpolated": self.is_interpolated
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
    def __init__(self, fps: float):
        self.brand_database = settings.BRAND_DATABASE
        self.fps = fps
        window_size = int(self.fps * settings.FRAME_WINDOW)
        self.brand_buffer = deque(maxlen=window_size)
        self.current_frame_number = 0
        self.video_resolution = None

    async def convert_relative_bbox(self, bbox: Dict) -> Dict:
        """
        Convert relative bounding box coordinates to absolute pixel values.
        """
        try:
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

    async def process_frame(self, frame: Frame) -> DetectionResult:
        self.current_frame_number = frame.number
        self.video_resolution = (frame.width, frame.height)

        logger.debug(f"Processing frame {frame.number}")

        text_detections = frame.ocr_data['rekognition_response'].get('TextDetections', [])
        logger.debug(f"Number of text detections: {len(text_detections)}")

        cleaned_detections = [await self.clean_detection(det) for det in text_detections]
        cleaned_detections = [det for det in cleaned_detections if det is not None]
        logger.debug(f"Cleaned detections: {len(cleaned_detections)} items")

        matched_brands = [await self.match_brand(det) for det in cleaned_detections]
        matched_brands = [brand for brand in matched_brands if brand is not None]

        # Group detections by brand
        brand_groups = {}
        for det in matched_brands:
            if det.text not in brand_groups:
                brand_groups[det.text] = []
            brand_groups[det.text].append(det)

        logger.debug(f"Frame {frame.number}: Matched brands before merging: {len(matched_brands)}")

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
                            merged_box = await self.merge_bounding_boxes(merged_detection.bounding_box, detections[j].bounding_box)
                            merged_detection = BrandInstance(
                                text=brand,
                                confidence=(merged_detection.confidence + detections[j].confidence) / 2,
                                bounding_box=merged_box,
                                original_text=f"{merged_detection.original_text} {detections[j].original_text}",
                                cleaned_text=f"{merged_detection.cleaned_text} {detections[j].cleaned_text}",
                                frame_number=frame.number,
                                is_interpolated=False
                            )
                            skip_indices.add(j)
                            merged = True
                    new_detections.append(merged_detection)
                if not merged:
                    break
                detections = new_detections
            merged_brands.extend(detections)

        logger.debug(f"Frame {frame.number}: Merged brands: {len(merged_brands)}")

        # Implement sliding window for consistency check
        window_size = int(self.fps * settings.FRAME_WINDOW)
        consistent_brands = []
        current_frame = frame.number

        for brand in merged_brands:
            # Count occurrences within the sliding window
            window_start = max(0, current_frame - window_size)
            window_end = current_frame
            
            occurrences = sum(1 for b in self.brand_buffer 
                            if b.text == brand.text 
                            and window_start <= b.frame_number <= window_end)
            
            if occurrences >= settings.MIN_DETECTIONS:
                consistent_brands.append(brand)
                logger.debug(f"Frame {current_frame}: Brand {brand.text} is consistent with {occurrences} occurrences in the last {window_size} frames")
            else:
                logger.debug(f"Frame {current_frame}: Brand {brand.text} not consistent enough (only {occurrences} occurrences in the last {window_size} frames)")

        # Update the brand buffer
        for brand in merged_brands:
            self.brand_buffer.append(brand)

        # Ensure brand_buffer doesn't exceed the window size
        while len(self.brand_buffer) > window_size:
            self.brand_buffer.popleft()

        logger.debug(f"Frame {frame.number}: Consistent brands: {len(consistent_brands)}")

        interpolated_brands = await self.interpolate_brands(frame.number, consistent_brands)
        logger.debug(f"Frame {frame.number}: Interpolated brands: {len(interpolated_brands)}")

        filtered_brands = await self.remove_over_interpolated_brands(interpolated_brands)
        logger.debug(f"Frame {frame.number}: Filtered brands: {len(filtered_brands)}")

        return DetectionResult(frame.number, filtered_brands)

    async def clean_detection(self, detection: Dict) -> Optional[Dict]:
        try:
            logger.debug(f"Cleaning detection: {detection['Type']} - {detection['DetectedText']}")
            
            if detection['Type'] != settings.OCR_TYPE:
                return None

            confidence = detection['Confidence']
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
            
            if brand_match and brand_score >= settings.LOW_CONFIDENCE_THRESHOLD:
                logger.debug(f"Frame {self.current_frame_number}: Matched brand: {brand_match} with score {brand_score}")
                return BrandInstance(
                    text=brand_match,
                    confidence=brand_score,
                    bounding_box=detection['bounding_box'],
                    original_text=detection['original_text'],
                    cleaned_text=detection['cleaned_text'],
                    frame_number=detection['frame_number']
                )
            return None
        except Exception as e:
            logger.error(f"Error in match_brand: {str(e)}", exc_info=True)
            return None

    async def interpolate_brands(self, frame_number: int, current_brands: List[BrandInstance]) -> List[BrandInstance]:
        interpolated_brands = []
        for brand in current_brands:
            self.brand_buffer.append(brand)
        
        for brand_name in set(brand.text for brand in self.brand_buffer):
            brand_instances = [b for b in self.brand_buffer if b.text == brand_name]
            interpolated_brand = await self.interpolate_brand(brand_name, frame_number, brand_instances)
            if interpolated_brand:
                interpolated_brands.append(interpolated_brand)

        return interpolated_brands

    async def interpolate_brand(self, brand_name: str, frame_number: int, instances: List[BrandInstance]) -> Optional[BrandInstance]:
        sorted_instances = sorted(instances, key=lambda x: x.frame_number)
        prev_instance = next((i for i in sorted_instances if i.frame_number < frame_number), None)
        next_instance = next((i for i in sorted_instances if i.frame_number > frame_number), None)

        if prev_instance and next_instance:
            return await self._interpolate_between(prev_instance, next_instance, frame_number)
        elif prev_instance:
            return await self._maintain_brand(prev_instance, frame_number)
        elif next_instance:
            return await self._fade_in_brand(next_instance, frame_number)
        else:
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

        # Interpolate confidence
        confidence = prev.confidence + t * (next.confidence - prev.confidence)
        
        return BrandInstance(
            text=prev.text,
            confidence=confidence,
            bounding_box=interpolated_box,
            original_text=prev.original_text,
            cleaned_text=prev.cleaned_text,
            frame_number=frame_number,
            is_interpolated=True
        )

    async def _maintain_brand(self, prev: BrandInstance, frame_number: int) -> BrandInstance:
        t = (frame_number - prev.frame_number) / settings.INTERPOLATION_LIMIT
        confidence_decay = np.exp(-0.5 * t)  # Slower decay
        
        return BrandInstance(
            text=prev.text,
            confidence=prev.confidence * confidence_decay,
            bounding_box=prev.bounding_box,
            original_text=prev.original_text,
            cleaned_text=prev.cleaned_text,
            frame_number=frame_number,
            is_interpolated=True
        )
    
    async def _fade_in_brand(self, next: BrandInstance, frame_number: int) -> BrandInstance:
        t = (frame_number - next.frame_number + settings.INTERPOLATION_LIMIT) / settings.INTERPOLATION_LIMIT
        confidence_increase = 1 - np.exp(-2 * t)  # Faster increase for fading in
        
        return BrandInstance(
            text=next.text,
            confidence=next.confidence * confidence_increase,
            bounding_box=next.bounding_box,
            original_text=next.original_text,
            cleaned_text=next.cleaned_text,
            frame_number=frame_number,
            is_interpolated=True
        )
    
    async def fuzzy_match_brand(self, text: str) -> Tuple[Optional[str], float]:
        best_match = None
        best_score = 0
        for brand in self.brand_database:
            score = fuzz.ratio(text.lower(), brand.lower())
            if score > best_score:
                best_score = score
                best_match = brand
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
            overlap_area = await BrandDetector.calculate_box_overlap(box1, box2)
            box1_area = BrandDetector.calculate_box_area(v1)
            box2_area = BrandDetector.calculate_box_area(v2)
            min_area = min(box1_area, box2_area)

            logger.debug(f"Overlap area: {overlap_area}, Box1 area: {box1_area}, Box2 area: {box2_area}")

            if overlap_area > 0 and overlap_area / min_area >= settings.MIN_OVERLAP_RATIO_FOR_MERGE:
                logger.debug("Boxes should be merged due to significant overlap")
                return True

            # Check if bottom of box1 is close to top of box2 or vice versa
            if close_edges(max(v1[2]['y'], v1[3]['y']), min(v2[0]['y'], v2[1]['y']), frame_height) or \
               close_edges(max(v2[2]['y'], v2[3]['y']), min(v1[0]['y'], v1[1]['y']), frame_height):
                logger.debug("Boxes should be merged due to close vertical edges")
                return True

            # Check if right of box1 is close to left of box2 or vice versa
            if close_edges(max(v1[1]['x'], v1[2]['x']), min(v2[0]['x'], v2[3]['x']), frame_width) or \
               close_edges(max(v2[1]['x'], v2[2]['x']), min(v1[0]['x'], v1[3]['x']), frame_width):
                logger.debug("Boxes should be merged due to close horizontal edges")
                return True

            logger.debug("Boxes should not be merged")
            return False

        except Exception as e:
            logger.error(f"Error in should_merge_bounding_boxes: {str(e)}", exc_info=True)
            return False  # Return False in case of error
        
    @staticmethod
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

        except Exception as e:
            logger.error(f"Error in merge_bounding_boxes: {str(e)}", exc_info=True)
            return box1  # Return first box in case of error

    @staticmethod
    def calculate_box_area(vertices):
        width = max(v['x'] for v in vertices) - min(v['x'] for v in vertices)
        height = max(v['y'] for v in vertices) - min(v['y'] for v in vertices)
        return width * height
########################################################