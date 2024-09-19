import time
import json
import traceback
import asyncio
import boto3
from typing import Tuple
from botocore.exceptions import ClientError
from core.logging import logger
from core.config import settings
from models.status_tracker import StatusTracker
from utils import utils
from services import s3_operations
from services.ocr_processing import ocr_cleaning, ocr_brand_matching

# Initialize Rekognition client
rekognition_client = boto3.client('rekognition', region_name=settings.AWS_DEFAULT_REGION)

## Runs OCR processing for an uploaded video
########################################################
async def process_ocr(video_id: str, status_tracker: 'StatusTracker', s3_client):
    logger.info(f"Starting OCR processing for video: {video_id}")
    status_tracker.update_process_status("ocr", "in_progress", 0)

    ocr_start_time = time.time()

    # Get video resolution
    try:
        video_resolution = await utils.get_video_resolution(video_id)
    except FileNotFoundError:
        logger.error(f"Cannot retrieve video resolution for OCR processing of video: {video_id}")
        status_tracker.set_error("Video resolution not found.")
        status_tracker.update_s3_status(s3_client)
        return
    except Exception as e:
        logger.error(f"Error retrieving video resolution for OCR processing of video {video_id}: {str(e)}")
        status_tracker.set_error("Error retrieving video resolution.")
        status_tracker.update_s3_status(s3_client)
        return

    # List frames in S3
    frame_objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    async for page in utils.async_paginate(paginator, Bucket=settings.PROCESSING_BUCKET, Prefix=f'{video_id}/frames/'):
        contents = page.get('Contents', [])
        frame_objects.extend(contents)

    total_frames = len(frame_objects)
    logger.info(f"Total frames to process for video {video_id}: {total_frames}")

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
                s3_client, 
                video_id, 
                int(frame['Key'].split('/')[-1].split('.')[0]), 
                video_resolution
            ) 
            for frame in batch
        ]
        batch_results = await asyncio.gather(*tasks)

        for processed_data, raw_data in batch_results:
            if processed_data and raw_data:
                ocr_results.append(processed_data)
                raw_ocr_results.append(raw_data)
                total_words += len(processed_data['full_text'].split())

        processed_frames += len(batch)
        progress = (processed_frames / total_frames) * 100
        status_tracker.update_process_status("ocr", "in_progress", progress)

        # Log progress periodically
        if processed_frames % 100 == 0 or processed_frames == total_frames:
            logger.info(f"Processed {processed_frames}/{total_frames} frames for video {video_id}")

    # Sort OCR results by frame number
    ocr_results.sort(key=lambda x: x['frame_number'])
    raw_ocr_results.sort(key=lambda x: x['frame_number'])

    # Store raw OCR results
    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=settings.PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/raw_ocr.json',
            Body=json.dumps(raw_ocr_results, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Saved raw OCR results for video: {video_id}")
    except ClientError as e:
        logger.error(f"Error saving raw OCR results for video {video_id}: {str(e)}")
        raise

    ocr_processing_time = time.time() - ocr_start_time
    frames_with_text = len([frame for frame in ocr_results if frame['text_annotations']])

    ocr_stats = {
        "ocr_processing_time": f"{ocr_processing_time:.2f} seconds",
        "frames_processed": total_frames,
        "frames_with_text": frames_with_text,
        "total_words_detected": total_words
    }

    status_tracker.update_process_status("ocr", "complete", 100)
    logger.info(f"Completed OCR processing for video: {video_id}")
    return ocr_stats
########################################################

## Perform OCR on a single video frame image
########################################################
async def process_single_frame_ocr(s3_client, video_id, frame_number, video_resolution: Tuple[int, int]):
    try:
        # Get the frame from S3
        response = await asyncio.to_thread(
            s3_client.get_object, 
            Bucket=settings.PROCESSING_BUCKET, 
            Key=f'{video_id}/frames/{frame_number:06d}.jpg'
        )
        image_content = response['Body'].read()

        # Perform OCR using Amazon Rekognition
        rekognition_response = rekognition_client.detect_text(
            Image={'Bytes': image_content}
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
                absolute_bbox = utils.convert_relative_bbox(relative_bbox, video_resolution)

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

            return processed_data, raw_data
        else:
            logger.info(f"No text found in frame {frame_number}")
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
        logger.error(f"Error processing OCR for frame {frame_number}: {str(e)}")
        return None, None
########################################################

## Post processing on raw OCR data
########################################################
async def post_process_ocr(video_id: str, fps: float, video_resolution: Tuple[int, int], s3_client):
    try:
        # Load OCR results
        ocr_results = s3_operations.load_ocr_results(s3_client, video_id)
        # logger.info(f"Loaded {len(ocr_results)} OCR results for video: {video_id}")

        # Step 1: Clean and consolidate OCR data
        # logger.info(f"Cleaning and consolidating OCR data for video: {video_id}")
        cleaned_results = ocr_cleaning.clean_and_consolidate_ocr_data(ocr_results, video_resolution)
        # logger.info(f"Cleaned and consolidated {len(cleaned_results)} frames for video: {video_id}")
        await s3_operations.save_processed_ocr_results(s3_client, video_id, cleaned_results)

        # Step 2: Create word cloud
        # logger.info(f"Creating word cloud for video: {video_id}")
        utils.create_word_cloud(s3_client, video_id, cleaned_results)

        # Step 3: Detect brands and interpolate
        # logger.info(f"Detecting brands and interpolating for video: {video_id}")
        brand_results, brand_appearances = ocr_brand_matching.detect_brands_and_interpolate(cleaned_results, fps, video_resolution)
        # logger.info(f"Detected {len(brand_appearances)} unique brands for video: {video_id}")

        # Step 4: Filter brand results
        # logger.info(f"Filtering brand results for video: {video_id}")
        filtered_brand_results = ocr_cleaning.filter_brand_results(brand_results, brand_appearances, fps)

        # Step 5: Save filtered brands OCR results
        await s3_operations.save_brands_ocr_results(s3_client, video_id, filtered_brand_results)

        # Step 6: Create and save brand table
        brand_stats = s3_operations.create_and_save_brand_table(s3_client, video_id, brand_appearances, fps)
        # logger.info(f"Created brand table with {len(brand_stats)} entries for video: {video_id}")

        logger.info(f"Completed post-processing OCR for video: {video_id}")
        return brand_stats
    except Exception as e:
        logger.error(f"Error in post_process_ocr for video {video_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise
########################################################