import io
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import Counter, defaultdict, OrderedDict
from wordcloud import WordCloud
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from utils.utils import find_font
from services import s3_operations
from models.detection_classes import DetectionResult, BrandDetector, Frame

## Task pool for parallel processing
########################################################
class TaskPool:
    def __init__(self, max_workers):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.tasks = set()

    async def run(self, coro):
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.tasks.add(task)
            try:
                return await task
            finally:
                self.tasks.remove(task)

    async def join(self):
        await asyncio.gather(*self.tasks)
########################################################

## Detect brands in raw OCR data
########################################################
async def detect_brands(
    video_id: str,
    status_tracker: StatusTracker,
    video_details: VideoDetails,
    ocr_results: Optional[List[Dict]] = None
) -> List[DetectionResult]:
    try:
        if ocr_results is None:
            logger.info(f"Video Processing - Brand Detection - Step 3.1: Loading OCR results for video: {video_id}")
            ocr_results = await s3_operations.load_ocr_results(video_id)
            logger.info(f"Video Processing - Brand Detection - Step 3.1: Loaded {len(ocr_results)} OCR results for video: {video_id}")

        video_width, video_height = await video_details.get_detail('video_resolution')
        video_fps = await video_details.get_detail('frames_per_second')
        brand_detector = BrandDetector(fps=video_fps, frame_width=video_width, frame_height=video_height)
        frames = [Frame(ocr['frame_number'], video_width, video_height, ocr) for ocr in ocr_results]

        total_frames = len(frames)
        logger.info(f"Video Processing - Brand Detection - Step 3.2: Starting to process {total_frames} frames for video: {video_id}")

        task_pool = TaskPool(settings.MAX_WORKERS)
        batch_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BATCHES)
        all_results = []

        async def process_batch(batch):
            async with batch_semaphore:
                return await task_pool.run(process_frame_batch(brand_detector, batch, status_tracker, total_frames))

        batches = [frames[i:i+settings.BATCH_SIZE] for i in range(0, total_frames, settings.BATCH_SIZE)]
        batch_tasks = [process_batch(batch) for batch in batches]
        
        for batch_result in asyncio.as_completed(batch_tasks):
            batch_results = await batch_result
            all_results.extend(batch_results)
            logger.info(f"Video Processing - Brand Detection - Step 3.2: Processed batch. Total frames processed: {len(all_results)}/{total_frames} ({((len(all_results)/total_frames) * 100):.0f}%)")

        await task_pool.join()

        # Step 3.3: Finalize brand detections
        logger.info(f"Video Processing - Brand Detection - Step 3.3: Finalizing brand detections for video: {video_id}")
        await post_process_interpolation(brand_detector)
        await remove_over_interpolated_brands(brand_detector)
        final_results = await finalize_brand_detections(brand_detector)
        logger.info(f"Video Processing - Brand Detection - Step 3.3: Finalized detection results for {len(final_results)} video frames")

        # Step 3.4: Validate results
        logger.info(f"Video Processing - Brand Detection - Step 3.4: Validating brand detection results for video: {video_id}")
        validated_results = [result for result in final_results if await validate_brand_result(result)]
        sorted_validated_results = sorted(validated_results, key=lambda x: x.frame_number)
        total_validated_brands = sum(len(result.detected_brands) for result in validated_results)
        logger.info(f"Video Processing - Brand Detection - Step 3.4: Validation complete. Total of {total_validated_brands} validated brand detections")

        # Step 3.5: Create word cloud
        logger.info(f"Video Processing - Brand Detection - Step 3.5: Creating brand word cloud for video: {video_id}")
        await create_word_cloud(video_id, sorted_validated_results)

        # Step 3.6: Save brand OCR results
        logger.info(f"Video Processing - Brand Detection - Step 3.6: Saving {total_validated_brands} brand detection results for video: {video_id}")
        await s3_operations.save_brands_ocr_results(video_id, sorted_validated_results)

        # Step 3.7: Create and save brand table
        logger.info(f"Video Processing - Brand Detection - Step 3.7: Creating and saving brand detections table for video: {video_id}")
        await create_brand_table(video_id, sorted_validated_results, video_fps)

        await status_tracker.update_process_status("ocr", "completed", 100)
        
        return sorted_validated_results

    except Exception as e:
        logger.error(f"Video Processing - Brand Detection: Error in detect_brands for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.update_process_status("ocr", "error", 0)
        raise
########################################################

## Process frames in batches
########################################################
async def process_frame_batch(brand_detector: BrandDetector, frames: List[Frame], status_tracker: StatusTracker, total_frames: int):
    results = []
    for frame in frames:
        result = await brand_detector.process_frame(frame)
        results.append(result)
        progress = int((frame.number + 1) / total_frames * 100)
        await status_tracker.update_process_status("ocr", "in_progress", progress)
    return results
########################################################

## xxx
########################################################
async def post_process_interpolation(brand_detector: BrandDetector):
    all_frames = sorted(set(b.frame_number for b in brand_detector.get_all_brand_detections()))
    all_brands = set(b.brand for b in brand_detector.get_all_brand_detections())
    
    for brand in all_brands:
        brand_frames = sorted(set(b.frame_number for b in brand_detector.get_all_brand_detections() if b.brand == brand))
        
        for i in range(len(brand_frames) - 1):
            start_frame = brand_frames[i]
            end_frame = brand_frames[i + 1]
            
            for frame in range(start_frame + 1, end_frame):
                if frame not in brand_frames:
                    interpolated_brand = await brand_detector.interpolate_brand(brand, frame)
                    if interpolated_brand:
                        brand_detector.add_brand_detection(interpolated_brand)
                        brand_detector.add_to_accumulated_results(frame, interpolated_brand)
    
    # Sort all_brand_detections by frame_number
    brand_detector.sort_all_brand_detections()
########################################################

## Remove over interpolated brands
########################################################
async def remove_over_interpolated_brands(brand_detector: BrandDetector):
    accumulated_results = brand_detector.get_accumulated_results()
    filtered_results = {}
    
    # Find the maximum frame number to ensure we process all frames
    max_frame = max(accumulated_results.keys()) if accumulated_results else 0
    
    for frame_number in range(max_frame + 1):
        brands = accumulated_results.get(frame_number, [])
        
        if not brands:
            # Preserve empty frames
            filtered_results[frame_number] = []
            continue
        
        filtered_brands = []
        for brand in brands:
            if not brand.is_interpolated:
                filtered_brands.append(brand)
            else:
                prev_frame = brand_detector.find_previous_non_interpolated_frame(brand.brand, frame_number)
                next_frame = brand_detector.find_next_non_interpolated_frame(brand.brand, frame_number)
                
                if prev_frame is not None and next_frame is not None:
                    if next_frame - prev_frame <= settings.INTERPOLATION_LIMIT:
                        filtered_brands.append(brand)
                    else:
                        logger.debug(f"Removed over-interpolated brand {brand.brand} at frame {frame_number}")
                else:
                    logger.debug(f"Removed interpolated brand {brand.brand} at frame {frame_number} without non-interpolated anchor")
        
        filtered_results[frame_number] = filtered_brands
    
    brand_detector.set_accumulated_results(filtered_results)
    logger.debug(f"Removed over-interpolated brands. Processed {len(filtered_results)} frames.")
########################################################

## Finalise brand detections
########################################################
async def finalize_brand_detections(brand_detector: BrandDetector) -> List[DetectionResult]:
    min_frames = max(1, int(settings.MIN_BRAND_TIME * brand_detector.fps))
    logger.debug(f"Minimum frames for brand appearance: {min_frames}")

    final_results = []
    brand_runs = defaultdict(list)
    current_run = defaultdict(list)

    # Identify runs of consecutive frames for each brand
    for frame_number, brands in sorted(brand_detector.get_accumulated_results().items()):
        for brand in brands:
            if frame_number - 1 not in current_run[brand.brand]:
                # Start of a new run
                if current_run[brand.brand]:
                    brand_runs[brand.brand].append(current_run[brand.brand])
                    current_run[brand.brand] = []
            current_run[brand.brand].append(frame_number)

    # Add any remaining runs
    for brand, run in current_run.items():
        if run:
            brand_runs[brand].append(run)

    # Filter out short runs
    for brand, runs in brand_runs.items():
        for run in runs:
            if len(run) < min_frames:
                logger.debug(f"Removing short run of brand '{brand}' from frames {run[0]} to {run[-1]}")
                for frame in run:
                    brand_detector.remove_brand_from_accumulated_results(frame, brand)
            else:
                logger.debug(f"Keeping run of brand '{brand}' from frames {run[0]} to {run[-1]}")

    # Create final results
    for frame_number, brands in sorted(brand_detector.get_accumulated_results().items()):
        final_results.append(DetectionResult(frame_number=frame_number, detected_brands=brands))

    return final_results
########################################################

## Create wordcloud
########################################################
async def create_word_cloud(video_id: str, results: List[DetectionResult]):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font. Only includes words with confidence above the minimum threshold
    and length greater than 2 characters.
    """
    
    # Extract all text annotations with confidence above the threshold and length > 2
    all_text = []
    excluded_confidence_count = 0
    excluded_length_count = 0
    brand_match_count = 0

    logger.debug(f"Video Processing - Brand Detection - Step 3.3 - OCR Data Processing: Processing cleaned OCR results for word cloud creation for video: {video_id}")
    for frame in results:
        for detection in frame.detected_brands:
            confidence = detection.brand_match_confidence
            
            # Use the brand field of BrandInstance
            text = detection.brand.lower().strip()
            
            if detection.is_interpolated:
                brand_match_count += 1

            if confidence >= settings.WORDCLOUD_MINIMUM_CONFIDENCE:
                if len(text) > 2:
                    all_text.append(text)
                else:
                    excluded_length_count += 1
            else:
                excluded_confidence_count += 1
    
    if not all_text:
        logger.warning(f"Video Processing - Brand Detection - Step 3.3: No text found for word cloud creation for video: {video_id}")
        return
    
    logger.debug(f"Video Processing - Brand Detection - Step 3.3: Excluded {excluded_confidence_count} words due to low confidence, "
                        f"{excluded_length_count} words due to short length. "
                        f"Included {brand_match_count} brand matches for video: {video_id}")

    # Count word frequencies
    word_freq = Counter(all_text)
    
    # Create a mask image (circle shape)
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # Find the first available preferred font
    font_path = await find_font(settings.PREFERRED_FONTS)
    if not font_path:
        logger.warning("Video Processing - Brand Detection - Step 3.3: No preferred font found. Using default.")

    try:
        logger.debug(f"Video Processing - Brand Detection - Step 3.3: Generating word cloud for video: {video_id}")
        # Generate word cloud
        wordcloud = WordCloud(width=600, height=600,
                                background_color='white',
                                max_words=100,  # Limit to top 100 words for clarity
                                min_font_size=10,
                                max_font_size=120,
                                mask=mask,
                                font_path=font_path,
                                colormap='ocean',  # Use a colorful colormap
                                prefer_horizontal=0.9,
                                scale=2
                                ).generate_from_frequencies(word_freq)

        # Create a figure and save it to a BytesIO object
        plt.figure(figsize=(10,10), frameon=False)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        img_buffer.seek(0)

        logger.debug(f"Video Processing - Brand Detection - Step 3.3: Uploading word cloud to S3 for video: {video_id}")
        
        # Upload to S3
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/wordcloud.jpg',
                Body=img_buffer.getvalue(),
                ContentType='image/jpeg'
            )
        logger.debug(f"Video Processing - Brand Detection - Step 3.3: Word cloud created and saved for video: {video_id}")
    except Exception as e:
        logger.error(f"Video Processing - Brand Detection - Step 3.3: Error creating or saving word cloud for video {video_id}: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close()
########################################################

## Create brand table
########################################################
async def create_brand_table(video_id: str, detection_results: List[DetectionResult], video_fps: float) -> Dict[str, Dict]:
    """
    Create a brand table with statistics for each detected brand, counting unique frames per brand.
    The frame_count will appear as the first item in each brand's statistics.

    :param video_id: The ID of the video being processed.
    :param detection_results: A list of DetectionResult objects containing brand detections.
    :param video_fps: The frames per second of the video.
    :return: A dictionary containing statistics for each detected brand, with frame_count first.
    """
    brand_stats = {}

    for result in detection_results:
        for brand_instance in result.detected_brands:
            brand = brand_instance.brand
            if brand not in brand_stats:
                brand_stats[brand] = {
                    "frames": set(),
                    "time_on_screen": 0,
                    "first_appearance": result.frame_number,
                    "last_appearance": result.frame_number,
                    "confidences": [],
                    "sizes": [],
                    "min_confidence": brand_instance.brand_match_confidence,
                    "max_confidence": brand_instance.brand_match_confidence
                }
            
            stats = brand_stats[brand]
            stats["frames"].add(result.frame_number)
            stats["time_on_screen"] += 1 / video_fps
            stats["last_appearance"] = max(stats["last_appearance"], result.frame_number)
            stats["confidences"].append(brand_instance.brand_match_confidence)
            stats["sizes"].append(brand_instance.relative_size)
            stats["min_confidence"] = min(stats["min_confidence"], brand_instance.brand_match_confidence)
            stats["max_confidence"] = max(stats["max_confidence"], brand_instance.brand_match_confidence)

    # Calculate averages, round values, and create final ordered dictionary with frame_count first
    final_brand_stats = {}
    for brand, stats in brand_stats.items():
        frame_count = len(stats["frames"])
        final_stats = OrderedDict([
            ("frame_count", frame_count),
            ("time_on_screen", round(stats["time_on_screen"], 2)),
            ("first_appearance", stats["first_appearance"]),
            ("last_appearance", stats["last_appearance"]),
            ("min_confidence", round(stats["min_confidence"], 2)),
            ("max_confidence", round(stats["max_confidence"], 2)),
            ("avg_confidence", round(sum(stats["confidences"]) / len(stats["confidences"]), 2)),
            ("avg_relative_size", round(sum(stats["sizes"]) / len(stats["sizes"]), 4))
        ])
        final_brand_stats[brand] = final_stats

    # Save brand stats to S3
    await s3_operations.create_and_save_brand_table(video_id, final_brand_stats)
    return final_brand_stats
########################################################

## Validate brand results
########################################################
async def validate_brand_result(result: DetectionResult) -> bool:
    valid = True
    for brand in result.detected_brands:
        if brand.is_interpolated:
            if brand.original_detected_text != "[INTERPOLATED]":
                logger.warning(f"Inconsistent data: Interpolated brand with non-interpolated original text in frame {result.frame_number}")
                valid = False
        else:
            if brand.original_detected_text == "[INTERPOLATED]":
                logger.warning(f"Inconsistent data: Non-interpolated brand with interpolated original text in frame {result.frame_number}")
                valid = False
        if brand.original_detected_text != "[INTERPOLATED]" and brand.original_detected_text_confidence < settings.MINIMUM_OCR_CONFIDENCE:
            logger.warning(f"Inconsistent data: Brand with low confidence in frame {result.frame_number}")
            valid = False
        logger.debug(f"Frame {result.frame_number}: Validated brand - {brand.brand}, Interpolated: {brand.is_interpolated}, Original text: {brand.original_detected_text}, Confidence: {brand.original_detected_text_confidence}")
    if not valid:
        logger.warning(f"Invalid result in frame {result.frame_number}")
    else:
        logger.debug(f"Valid result in frame {result.frame_number} with {len(result.detected_brands)} brands")
    return valid
########################################################