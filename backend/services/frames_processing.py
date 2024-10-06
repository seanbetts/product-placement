import os
import time
import cv2
import re
import asyncio
from typing import List
from io import BytesIO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from core.config import settings
from core.logging import logger
from services import s3_operations
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from utils.utils import get_video_resolution
from core.aws import get_s3_client
from functools import lru_cache
from bisect import insort

# Create a cache for frame data
@lru_cache(maxsize=100)
async def get_cached_frame(frame_key):
    async with get_s3_client() as s3_client:
        response = await s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=frame_key)
    data = await response['Body'].read()
    return data

## Process video frames
########################################################
async def process_video_frames(video_path: str, video_id: str, status_tracker: StatusTracker, video_details: VideoDetails):
    try:
        logger.info(f"Video Processing - Thread 1 - Image Processing - Step 1.1: Started video frame extraction for video: {video_id}")
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)

        # Set file size
        file_size = os.path.getsize(video_path)
        logger.debug(f"Video File Size: {(file_size / 1000000):.1f} Mb")
        video_details.set_detail("file_size", file_size)
        
        # Set FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.debug(f"FPS: {fps}")
        video_details.set_detail("frames_per_second", fps)

        # Set number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"Frame Count: {frame_count}")
        video_details.set_detail("number_of_frames", frame_count)

        # Set video length
        duration = frame_count / fps
        logger.debug(f"Duration: {duration:.2f} seconds")
        video_details.set_detail("video_length", duration)
        
        logger.debug(f"Video details - File size: {(file_size / 1000000):.1f} Mb, FPS: {fps:.1f}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")
        
        frame_number = 0
        tasks = []
        current_batch = []
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BATCHES)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if current_batch:
                    tasks.append(asyncio.create_task(process_batch(current_batch.copy(), video_id, semaphore)))
                break
            if frame_number % settings.FRAME_INTERVAL == 0:
                success, frame_data = cv2.imencode('.jpg', frame)
                if not success:
                    logger.error(f"Failed to encode frame {frame_number}")
                    continue
                frame_bytes = frame_data.tobytes()
                frame_filename = f'{frame_number:06d}.jpg'
                current_batch.append((frame_bytes, frame_filename))
                if len(current_batch) == settings.BATCH_SIZE:
                    tasks.append(asyncio.create_task(process_batch(current_batch.copy(), video_id, semaphore)))
                    current_batch = []
            
            frame_number += 1
            
            # Update status every 5% of frames processed
            if frame_number % max(1, frame_count // 20) == 0:
                progress = (frame_number / frame_count) * 100
                await status_tracker.update_process_status("video_processing", "in_progress", progress)
        
        cap.release()
        logger.debug(f"Video Processing - Thread 1 - Image Processing - Step 1.1: Finished reading video. Total batches: {len(tasks)}")
        
        # Wait for all batch processing tasks to complete
        total_frames_processed = sum(await asyncio.gather(*tasks))
        
        logger.debug(f"Video Processing - Thread 1 - Image Processing - Step 1.1: Total frames processed and uploaded: {total_frames_processed}")

        # Set video resolution
        video_resolution = await get_video_resolution(video_id)
        logger.debug(f"Video Processing - Thread 1 - Image Processing - Step 1.1: Video Resolution: {video_resolution}")
        video_details.set_detail("video_resolution", video_resolution)
        
        processing_time = time.time() - start_time
        extracted_frames = frame_number // settings.FRAME_INTERVAL
        processing_fps = extracted_frames / processing_time
        processing_speed = (processing_fps / fps) * 100
        
        result = {
            "video_length": f'{duration:.1f} seconds',
            "total_frames": frame_count,
            "extracted_frames": extracted_frames,
            "video_fps": round(fps, 2),
            "video_processing_time": processing_time,
            "video_processing_fps": processing_fps,
            "video_processing_speed": processing_speed
        }
        
        logger.debug(f"Video processing completed. Results: {result}")
        logger.info(f"Video Processing - Thread 1 - Image Processing - Step 1.2: Video frame extraction for video: {video_id} completed with {total_frames_processed} frames processed")
        await status_tracker.update_process_status("video_processing", "complete", 100)
        return result
    
    except Exception as e:
        logger.error(f"Video Processing - Thread 1 - Image Processing - Error in video frame processing: {str(e)}")
        await status_tracker.set_error(f"Video Processing - Thread 1 - Image Processing - Video frame processing error: {str(e)}")
        raise
########################################################

## Process video frame image batches
########################################################
async def process_batch(batch, video_id, semaphore):
    async with semaphore:
        try:
            logger.debug(f"Starting to process batch for video {video_id} with {len(batch)} frames")
            successful_uploads = 0

            upload_tasks = []
            for frame_bytes, frame_filename in batch:
                s3_key = f'{video_id}/frames/{frame_filename}'
                upload_tasks.append(s3_operations.upload_frame_to_s3(s3_key, frame_bytes))

            results = await asyncio.gather(*upload_tasks)
            successful_uploads = sum(results)

            logger.debug(f"Successfully uploaded {successful_uploads} out of {len(batch)} frames for video {video_id}")
            
            if successful_uploads != len(batch):
                logger.warning(f"Failed to upload {len(batch) - successful_uploads} frames for video {video_id}")
            
            return successful_uploads
        
        except Exception as e:
            logger.error(f"Error processing batch for video {video_id}: {str(e)}", exc_info=True)
            return 0
########################################################

## Get first video frame
########################################################
async def get_first_video_frame(video_id: str):
    # Use cached frame if available
    first_frame_path = f'{video_id}/frames/000000.jpg'

    try:
        frame_data = await get_cached_frame(first_frame_path)
        return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
    except Exception:
        # If not in cache, fetch from S3
        try:
            async with get_s3_client() as s3_client:
                response = await s3_client.get_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=first_frame_path
                )
            frame_data = await response['Body'].read()
            return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
        except Exception as e:
            logger.error(f"Error retrieving first frame for video {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=404, detail="First frame not found")
########################################################

## Get all video frames
########################################################
async def get_all_video_frames(video_id: str) -> List[dict]:
    """
    Retrieve information about processed frames for a given video.

    Args:
        video_id (str): The ID of the video.

    Returns:
        List[dict]: A list of dictionaries containing frame information.
        Each dictionary contains 'number' (frame number) and 'url' (signed S3 URL).

    Raises:
        HTTPException: If there's an error processing the frames.
    """
    FRAME_FILENAME_PREFIX = "processed_frame_"
    FRAME_FILENAME_PATTERN = re.compile(rf"{FRAME_FILENAME_PREFIX}(\d+)\.jpg")
    frames_prefix = f'{video_id}/processed_frames/'
    frames = []
    total_frames_processed = 0
    frames_returned = 0

    try:
        async with get_s3_client() as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=settings.PROCESSING_BUCKET, Prefix=frames_prefix):
                for obj in page.get('Contents', []):
                    total_frames_processed += 1
                    try:
                        filename = obj['Key'].split('/')[-1]
                        match = FRAME_FILENAME_PATTERN.match(filename)
                        if match:
                            frame_number = int(match.group(1))
                            if frame_number % 50 == 0:
                                signed_url = await s3_client.generate_presigned_url(
                                    'get_object',
                                    Params={'Bucket': settings.PROCESSING_BUCKET, 'Key': obj['Key']},
                                    ExpiresIn=3600
                                )
                                frame_info = {
                                    "number": frame_number,
                                    "url": signed_url
                                }
                                insort(frames, frame_info, key=lambda x: x["number"])
                                frames_returned += 1
                        else:
                            logger.warning(f"Skipping file with unexpected format: {filename}")
                    except Exception as e:
                        logger.error(f"Error processing frame {obj['Key']}: {str(e)}")

        logger.debug(f"Processed {total_frames_processed} frames, returning {frames_returned} frames for video {video_id}")
        return frames
    except Exception as e:
        logger.error(f"Error processing frames for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")
########################################################