import time
import json
import asyncio
import boto3
from typing import Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from utils import utils
from utils.decorators import retry
from services import s3_operations
from services.ocr_processing import ocr_cleaning, ocr_brand_matching

# Initialize Rekognition client
rekognition_client = boto3.client('rekognition', region_name=settings.AWS_DEFAULT_REGION)

# Create a thread pool executor
thread_pool = ThreadPoolExecutor()

## Runs OCR processing for an uploaded video
########################################################
async def process_ocr(video_id: str, status_tracker: 'StatusTracker', video_details: VideoDetails) -> Dict:
    logger.info(f"Video Processing - Thread 1 - Image Processing - Step 2.2: Detecting text in video frames of video: {video_id}")
    ocr_start_time = time.time()

    video_resolution = await video_details.get_detail("video_resolution")

    try:
            # Asynchronous S3 listing
        async def list_objects_async(prefix):
            all_objects = []
            async with get_s3_client() as s3_client:
                paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=settings.PROCESSING_BUCKET, Prefix=prefix):
                all_objects.extend(page.get('Contents', []))
                logger.debug(f"Retrieved {len(all_objects)} objects so far for video: {video_id}")
            return all_objects

        logger.debug(f"Starting S3 listing for video: {video_id}")
        frame_objects = await list_objects_async(f'{video_id}/frames/')
        logger.debug(f"Completed S3 listing for video: {video_id}. Total frames: {len(frame_objects)}")

        total_frames = len(frame_objects)
        logger.debug(f"Total frames to process for video {video_id}: {total_frames}")

        ocr_results = []
        raw_ocr_results = []
        processed_frames = 0
        total_words = 0

        # Process frames in batches to limit concurrency
        batch_size = int(settings.BATCH_SIZE)
        for i in range(0, len(frame_objects), batch_size):
            batch = frame_objects[i:i+batch_size]
            tasks = [
                process_single_frame_ocr(
                    video_id,
                    int(frame['Key'].split('/')[-1].split('.')[0]),
                    video_resolution
                )
                for frame in batch
            ]
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Video Processing - Thread 1 - Image Processing - Step 2.2: Error processing frame batch for video {video_id}: {str(e)}", exc_info=True)
                continue

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Video Processing - Thread 1 - Image Processing - Step 2.2: Error processing frame for video {video_id}: {str(result)}")
                    continue
                processed_data, raw_data = result
                if processed_data and raw_data:
                    ocr_results.append(processed_data)
                    raw_ocr_results.append(raw_data)
                    total_words += len(processed_data['full_text'].split())

            processed_frames += len(batch)
            progress = (processed_frames / total_frames) * 80  # Cap at 80%
            await status_tracker.update_process_status("ocr", "in_progress", min(progress, 80))

            # Log progress periodically
            if processed_frames % 100 == 0 or processed_frames == total_frames:
                logger.debug(f"Processed {processed_frames}/{total_frames} frames for video {video_id}")
                continue

        # Sort OCR results by frame number
        ocr_results.sort(key=lambda x: x['frame_number'])
        raw_ocr_results.sort(key=lambda x: x['frame_number'])

        # Store raw OCR results
        @retry(exceptions=(ClientError), tries=3, delay=1, backoff=2)
        async def upload_raw_ocr_results(s3_client, video_id, raw_ocr_results):
            raw_ocr_json = json.dumps(raw_ocr_results, indent=2)
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/raw_ocr.json',
                Body=raw_ocr_json,
                ContentType='application/json'
            )
            return len(raw_ocr_json)

        try:
            async with get_s3_client() as s3_client:
                uploaded_size = await upload_raw_ocr_results(s3_client, video_id, raw_ocr_results)
            logger.info(f"Video Processing - Thread 1 - Image Processing - Step 2.3: Saved raw OCR results for video: {video_id}")
        except Exception as e:
            logger.error(f"Video Processing - Thread 1 - Image Processing - Step 2.3: Error saving raw OCR results for video {video_id}: {str(e)}", exc_info=True)
            await status_tracker.update_process_status("ocr", "error", 80)
            await status_tracker.set_error(f"Video Processing - Thread 1 - Image Processing - Step 2.3: Error saving raw OCR results: {str(e)}")
            return None

        ocr_processing_time = time.time() - ocr_start_time
        frames_with_text = len([frame for frame in ocr_results if frame['text_annotations']])
        ocr_stats = {
            "ocr_processing_time": f"{ocr_processing_time:.2f} seconds",
            "frames_processed": total_frames,
            "frames_with_text": frames_with_text,
            "total_words_detected": total_words
        }

        await status_tracker.update_process_status("ocr", "in_progress", 80)  # Ensure we're at 80% before post-processing
        return ocr_stats

    except Exception as e:
        logger.error(f"Video Processing - Thread 1 - Image Processing - Step 2.4: Error in OCR processing for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.update_process_status("ocr", "error", 80)
        await status_tracker.set_error(f"Video Processing - Thread 1 - Image Processing - Step 2.4: Error in OCR processing: {str(e)}")
        return None
########################################################

## Perform OCR on a single video frame image
########################################################
async def process_single_frame_ocr(video_id: str, frame_number: int, video_resolution: Tuple[int, int]):
    try:
        # Get the frame from S3
        logger.debug(f"Retrieving frame {frame_number} for video {video_id}")
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/frames/{frame_number:06d}.jpg'
            )
        image_content = await response['Body'].read()

        # Perform OCR using Amazon Rekognition in a separate thread
        logger.debug(f"Performing OCR on frame {frame_number} for video {video_id}")
        loop = asyncio.get_running_loop()
        rekognition_response = await loop.run_in_executor(
            thread_pool,
            lambda: rekognition_client.detect_text(Image={'Bytes': image_content})
        )

        # Process the Rekognition response
        texts = rekognition_response.get('TextDetections', [])
        if texts:
            # Concatenate all detected text snippets
            full_text = ' '.join([text['DetectedText'] for text in texts if text['Type'] == 'LINE'])
            
            # Process individual text detections
            text_annotations = []
            for text in texts:
                if text['Type'] != 'LINE':
                    continue  # Skip non-line detections
                relative_bbox = text['Geometry']['BoundingBox']
                absolute_bbox = await utils.convert_relative_bbox(relative_bbox, video_resolution)
                text_annotation = {
                    "text": text['DetectedText'],
                    "bounding_box": absolute_bbox
                }
                text_annotations.append(text_annotation)

            processed_data = {
                "frame_number": frame_number,
                "full_text": full_text,
                "text_annotations": text_annotations
            }
            raw_data = {
                "frame_number": frame_number,
                "rekognition_response": rekognition_response
            }
            logger.debug(f"Processed OCR data for frame {frame_number} of video {video_id}")
            return processed_data, raw_data
        else:
            logger.debug(f"No text found in frame {frame_number} of video {video_id}")
            processed_data = {
                "frame_number": frame_number,
                "full_text": "",
                "text_annotations": []
            }
            raw_data = {
                "frame_number": frame_number,
                "rekognition_response": rekognition_response
            }
            return processed_data, raw_data

    except Exception as e:
        logger.error(f"Video Processing - Thread 1 - Image Processing: Error processing OCR for frame {frame_number} of video {video_id}: {str(e)}", exc_info=True)
        return None, None
########################################################

## Post processing on raw OCR data
########################################################
async def brand_detection(video_id: str, status_tracker: StatusTracker, video_details: VideoDetails):
    try:
        # Assume post-processing is 20% of the total OCR process
        total_steps = 7
        step_progress = 20 / total_steps

        video_resolution = await video_details.get_detail('video_resolution')
        fps = await video_details.get_detail('frames_per_second')

        await status_tracker.update_process_status("ocr", "in_progress", 80)  # Start at 80%

        # Load OCR results
        logger.info(f"Video Processing - Brand Detection - Step 1.2: Loading OCR results for video: {video_id}")
        ocr_results = await s3_operations.load_ocr_results(video_id)
        logger.debug(f"Loaded {len(ocr_results)} OCR results for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 80 + step_progress)

        # Step 1: Clean and consolidate OCR data
        logger.info(f"Video Processing - Brand Detection - Step 2.1: Cleaning and consolidating OCR data for video: {video_id}")
        cleaned_results = await ocr_cleaning.clean_and_consolidate_ocr_data(ocr_results, video_resolution)
        logger.info(f"Video Processing - Brand Detection - Step 2.4: Cleaned and consolidated {len(cleaned_results)} frames for video: {video_id}")
        logger.info(f"Video Processing - Brand Detection - Step 2.5: Saving processed OCR results for video: {video_id}")
        await s3_operations.save_processed_ocr_results(video_id, cleaned_results)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 2 * step_progress)

        # Step 2: Create word cloud
        logger.info(f"Video Processing - Brand Detection - Step 3.1: Creating word cloud for video: {video_id}")
        await utils.create_word_cloud(video_id, cleaned_results)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 3 * step_progress)

        # Step 3: Detect brands and interpolate
        logger.info(f"Video Processing - Brand Detection - Step 4.1: Detecting brands and interpolating for video: {video_id}")
        brand_results, brand_appearances = await ocr_brand_matching.detect_brands_and_interpolate(cleaned_results, fps, video_resolution)
        logger.info(f"Video Processing - Brand Detection - Step 4.5: Detected {len(brand_appearances)} unique brands for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 4 * step_progress)

        # Step 4: Filter brand results
        logger.info(f"Video Processing - Brand Detection - Step 5.1: Filtering brand results for video: {video_id}")
        filtered_brand_results = await ocr_cleaning.filter_brand_results(brand_results, brand_appearances, fps)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 5 * step_progress)

        # Step 5: Save filtered brands OCR results
        logger.info(f"Video Processing - Brand Detection - Step 6.1: Saving filtered brand OCR results for video: {video_id}")
        await s3_operations.save_brands_ocr_results(video_id, filtered_brand_results)
        await status_tracker.update_process_status("ocr", "in_progress", 80 + 6 * step_progress)

        # Step 6: Create and save brand table
        logger.info(f"Video Processing - Brand Detection - Step 7.1: Creating and saving brand table for video: {video_id}")
        brand_stats = await s3_operations.create_and_save_brand_table(video_id, brand_appearances, fps)
        logger.info(f"Video Processing - Brand Detection - Step 7.2: Created brand table with {len(brand_stats)} entries for video: {video_id}")
        await status_tracker.update_process_status("ocr", "in_progress", 100)

        logger.info(f"Video Processing - Brand Detection: Completed post-processing OCR for video: {video_id}")
        return brand_stats
    except Exception as e:
        logger.error(f"Video Processing - Brand Detection: Error in post_process_ocr for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.update_process_status("ocr", "error", 80)
        await status_tracker.set_error(f"Video Processing - Brand Detection: Error in post-OCR processing: {str(e)}")
        raise
########################################################