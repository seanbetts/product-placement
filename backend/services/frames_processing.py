import os
import time
import cv2
import tempfile
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List
from io import BytesIO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from core.config import settings
from core.logging import AppLogger
from services import s3_operations
from models.status_tracker import StatusTracker
from core.aws import get_s3_client
from functools import lru_cache

# Create a global instance of AppLogger
app_logger = AppLogger()

# Create a cache for frame data
@lru_cache(maxsize=100)
async def get_cached_frame(frame_key):
    s3_client = await get_s3_client()
    response = await s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=frame_key)
    data = await response['Body'].read()
    return data

## Process video frames
########################################################
async def process_video_frames(vlogger, video_path: str, video_id: str, status_tracker: StatusTracker):
    @vlogger.log_performance
    async def _process_video_frames():
        try:
            start_time = time.time()
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            vlogger.logger.info(f"Video details - FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")
            
            frame_number = 0
            tasks = []
            current_batch = []
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_BATCHES)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    if current_batch:
                        tasks.append(asyncio.create_task(process_batch(vlogger, current_batch.copy(), video_id, semaphore)))
                    break
                if frame_number % settings.FRAME_INTERVAL == 0:
                    success, frame_data = cv2.imencode('.jpg', frame)
                    if not success:
                        vlogger.logger.error(f"Failed to encode frame {frame_number}")
                        continue
                    frame_bytes = frame_data.tobytes()
                    frame_filename = f'{frame_number:06d}.jpg'
                    current_batch.append((frame_bytes, frame_filename))
                    if len(current_batch) == settings.BATCH_SIZE:
                        tasks.append(asyncio.create_task(process_batch(vlogger, current_batch.copy(), video_id, semaphore)))
                        current_batch = []
                
                frame_number += 1
                
                # Update status every 5% of frames processed
                if frame_number % max(1, frame_count // 20) == 0:
                    progress = (frame_number / frame_count) * 100
                    await status_tracker.update_process_status("video_processing", "in_progress", progress)
            
            cap.release()
            vlogger.logger.info(f"Finished reading video. Total batches: {len(tasks)}")
            
            # Wait for all batch processing tasks to complete
            total_frames_processed = sum(await asyncio.gather(*tasks))
            
            vlogger.logger.info(f"Total frames processed and uploaded: {total_frames_processed}")
            
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
            
            vlogger.logger.info(f"Video processing completed. Results: {result}")
            await status_tracker.update_process_status("video_processing", "complete", 100)
            return result
        
        except Exception as e:
            vlogger.logger.error(f"Error in video frame processing: {str(e)}", exc_info=True)
            await status_tracker.set_error(f"Video frame processing error: {str(e)}")
            raise

    return await _process_video_frames()
########################################################

## Process video frame image batches
########################################################
async def process_batch(vlogger, batch, video_id, semaphore):
    @vlogger.log_performance
    async def _process_batch():
        async with semaphore:
            try:
                s3_client = await get_s3_client()
                vlogger.logger.info(f"Starting to process batch for video {video_id} with {len(batch)} frames")
                successful_uploads = 0

                upload_tasks = []
                for frame_bytes, frame_filename in batch:
                    s3_key = f'{video_id}/frames/{frame_filename}'
                    upload_tasks.append(s3_operations.upload_frame_to_s3(vlogger, settings.PROCESSING_BUCKET, s3_key, frame_bytes))

                results = await asyncio.gather(*upload_tasks)
                successful_uploads = sum(results)

                vlogger.logger.info(f"Successfully uploaded {successful_uploads} out of {len(batch)} frames for video {video_id}")
                
                if successful_uploads != len(batch):
                    vlogger.logger.warning(f"Failed to upload {len(batch) - successful_uploads} frames for video {video_id}")
                
                return successful_uploads
            
            except Exception as e:
                vlogger.logger.error(f"Error processing batch for video {video_id}: {str(e)}", exc_info=True)
                return 0

    return await _process_batch()

########################################################

## Get first video frame
########################################################
async def get_first_video_frame(video_id: str):
    # Use cached frame if available
    first_frame_path = f'{video_id}/frames/000000.jpg'

    s3_client = await get_s3_client()

    try:
        frame_data = await get_cached_frame(first_frame_path)
        return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
    except Exception:
        # If not in cache, fetch from S3
        try:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=first_frame_path
            )
            frame_data = await response['Body'].read()
            return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
        except Exception as e:
            app_logger.log_error(f"Error retrieving first frame for video {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=404, detail="First frame not found")
########################################################

## Get all video frames
########################################################
async def get_all_video_frames(video_id: str) -> List[dict]:
    frames_prefix = f'{video_id}/frames/'

    s3_client = await get_s3_client()
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # paginate() is not async, but you can iterate over it in an async loop if needed
        async for page in paginator.paginate(Bucket=settings.PROCESSING_BUCKET, Prefix=frames_prefix):
            frames = []
        
            for obj in page.get('Contents', []):
                try:
                    frame_number = int(obj['Key'].split('/')[-1].split('.')[0])
                    if frame_number % 50 == 0:
                        # Generate a pre-signed URL for the frame
                        signed_url = await s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': settings.PROCESSING_BUCKET, 'Key': obj['Key']},
                            ExpiresIn=3600
                        )
                        frames.append({
                            "number": frame_number,
                            "url": signed_url
                        })
                except Exception as e:
                    app_logger.log_error(f"Error generating signed URL for object {obj['Key']}: {str(e)}")
        
        sorted_frames = sorted(frames, key=lambda x: x["number"])
        return sorted_frames

    except Exception as e:
        app_logger.log_error(f"Error processing frames for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")
########################################################