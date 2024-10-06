import boto3
import asyncio
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor
from core.config import settings
from core.logging import logger
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from utils import utils
from services import s3_operations
from services.ocr_processing import ocr_cleaning, ocr_brand_matching

# Initialize Rekognition client
rekognition_client = boto3.client('rekognition', region_name=settings.AWS_DEFAULT_REGION)

# Create a thread pool executor
thread_pool = ThreadPoolExecutor()

## Detect brands in raw OCR data
########################################################
async def detect_brands(video_id: str, status_tracker: StatusTracker, video_details: VideoDetails):
    try:
        # Assume detecting brands is 20% of the total OCR process
        total_steps = 7
        step_progress = 20 / total_steps

        video_resolution = await video_details.get_detail('video_resolution')
        fps = await video_details.get_detail('frames_per_second')

        await status_tracker.update_process_status("ocr", "in_progress", 80)  # Start at 80%

        # Load OCR results
        logger.info(f"Video Processing - Brand Detection - Step 3.1.2: Loading OCR results for video: {video_id}")
        ocr_results = await s3_operations.load_ocr_results(video_id)
        logger.debug(f"Loaded {len(ocr_results)} OCR results for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 80 + step_progress)

        # Step 1: Clean and consolidate OCR data
        logger.info(f"Video Processing - Brand Detection - Step 3.2.1: Cleaning and consolidating OCR data for video: {video_id}")
        cleaned_results = await ocr_cleaning.clean_and_consolidate_ocr_data(ocr_results, video_resolution)
        logger.info(f"Video Processing - Brand Detection - Step 3.2.4: Cleaned and consolidated {len(cleaned_results)} frames for video: {video_id}")
        await s3_operations.save_processed_ocr_results(video_id, cleaned_results)
        logger.info(f"Video Processing - Brand Detection - Step 3.2.5: Saved processed OCR results for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 2 * step_progress)

        # Step 2: Create word cloud
        logger.info(f"Video Processing - Brand Detection - Step 3.3.1: Creating word cloud for video: {video_id}")
        await utils.create_word_cloud(video_id, cleaned_results)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 3 * step_progress)

        # Step 3: Detect brands and interpolate
        logger.info(f"Video Processing - Brand Detection - Step 3.4.1: Detecting brands and interpolating for video: {video_id}")
        brand_results, brand_appearances = await ocr_brand_matching.detect_brands_and_interpolate(cleaned_results, fps, video_resolution)
        logger.info(f"Video Processing - Brand Detection - Step 3.4.5: Detected {len(brand_appearances)} unique brands for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 4 * step_progress)

        # Step 4: Filter brand results
        logger.info(f"Video Processing - Brand Detection - Step 3.5.1: Filtering brand results for video: {video_id}")
        filtered_brand_results = await filter_brand_results(brand_results, brand_appearances, fps)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 5 * step_progress)

        # Step 5: Save filtered brands OCR results
        logger.info(f"Video Processing - Brand Detection - Step 3.6.1: Saving filtered brand OCR results for video: {video_id}")
        await s3_operations.save_brands_ocr_results(video_id, filtered_brand_results)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 6 * step_progress)

        # Step 6: Create and save brand table
        logger.info(f"Video Processing - Brand Detection - Step 3.7.1: Creating and saving brand table for video: {video_id}")
        brand_stats = await s3_operations.create_and_save_brand_table(video_id, brand_appearances, fps)
        logger.info(f"Video Processing - Brand Detection - Step 3.7.2: Created brand table with {len(brand_stats)} entries for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 100)

        logger.info(f"Video Processing - Brand Detection: Completed post-processing OCR for video: {video_id}")
        return filtered_brand_results
    except Exception as e:
        logger.error(f"Video Processing - Brand Detection: Error in post_process_ocr for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.update_process_status("ocr", "error", 80)
        await status_tracker.set_error(f"Video Processing - Brand Detection: Error in post-OCR processing: {str(e)}")
        raise
########################################################

## Filter brand results from OCR
########################################################
async def filter_brand_results(brand_results: List[Dict], brand_appearances: Dict[str, Set[int]], fps: float) -> List[Dict]:
    try:
        logger.debug("Video Processing - Brand Detection - Step 5.1: Starting to filter brand results")

        min_frames = int(fps * settings.MIN_BRAND_TIME)
        logger.debug(f"Video Processing - Brand Detection - Step 5.1: Minimum frames for a valid brand: {min_frames}")

        valid_brands = {brand for brand, appearances in brand_appearances.items() if len(appearances) >= min_frames}
        logger.debug(f"Video Processing - Brand Detection - Step 5.1: Found {len(valid_brands)} valid brands out of {len(brand_appearances)} total brands")

        async def process_frame(frame):
            filtered_brands = [brand for brand in frame['detected_brands'] if brand['text'] in valid_brands]
            return {
                "frame_number": frame['frame_number'],
                "detected_brands": filtered_brands
            }

        filtered_results = await asyncio.gather(*[process_frame(frame) for frame in brand_results])

        total_brands_before = sum(len(frame['detected_brands']) for frame in brand_results)
        total_brands_after = sum(len(frame['detected_brands']) for frame in filtered_results)

        logger.debug(f"Video Processing - Brand Detection - Step 5.1: Filtering complete. Brands reduced from {total_brands_before} to {total_brands_after}")
        return filtered_results

    except Exception as e:
        logger.error(f"Video Processing - Brand Detection: Error in filter_brand_results: {str(e)}", exc_info=True)
        return brand_results  # Return original results in case of error
########################################################