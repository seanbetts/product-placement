import cv2
import time
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List
from io import BytesIO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from core.config import settings
from core.logging import AppLogger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from services import s3_operations

# Create a global instance of AppLogger
app_logger = AppLogger()

# Create a global instance of s3_client
s3_client = get_s3_client()

## Process video frames
########################################################
async def process_video_frames(vlogger, video_path: str, video_id: str, s3_client, status_tracker: StatusTracker):
    @vlogger.log_performance
    async def _process_video_frames():
        try:
            start_time = time.time()
            vlogger.logger.info(f"Starting to process video frames for video ID: {video_id}")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            vlogger.logger.info(f"Video details - FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")
            
            frame_number = 0
            batches = []
            current_batch = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    if current_batch:
                        batches.append(current_batch)
                    break
                
                if frame_number % settings.FRAME_INTERVAL == 0:
                    current_batch.append((frame, frame_number))
                    if len(current_batch) == settings.BATCH_SIZE:
                        batches.append(current_batch)
                        current_batch = []
                
                frame_number += 1
                progress = (frame_number / frame_count) * 100
                await status_tracker.update_process_status("video_processing", "in_progress", progress)
            
            cap.release()
            vlogger.logger.info(f"Finished reading video. Total batches: {len(batches)}")
            
            # Process batches in parallel
            vlogger.logger.info(f"Starting parallel processing of {len(batches)} batches")
            total_frames_processed = 0
            
            async def process_batch_wrapper(batch):
                try:
                    return await process_batch(vlogger, batch, video_id, s3_client)
                except Exception as e:
                    vlogger.logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                    await status_tracker.set_error(f"Batch processing error: {str(e)}")
                    return 0
            
            with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
                loop = asyncio.get_event_loop()
                futures = [loop.run_in_executor(executor, lambda b=batch: asyncio.run(process_batch_wrapper(b))) for batch in batches]
                results = await asyncio.gather(*futures)
            
            total_frames_processed = sum(results)
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

## Process batches of frames
########################################################
async def process_batch(vlogger, batch, video_id, s3_client):
    try:
        vlogger.logger.info(f"Starting to process batch for video {video_id} with {len(batch)} frames")
        frames = []
        for frame, frame_number in batch:
            frame_filename = f'{frame_number:06d}.jpg'
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append((f'{video_id}/frames/{frame_filename}', buffer.tobytes()))

        vlogger.logger.debug(f"Initiating batch upload for {len(frames)} frames")
        successful_uploads = await s3_operations.upload_frames_batch(vlogger, s3_client, settings.PROCESSING_BUCKET, frames)

        vlogger.logger.info(f"Successfully uploaded {successful_uploads} out of {len(frames)} frames for video {video_id}")
        
        if successful_uploads != len(frames):
            vlogger.logger.warning(f"Failed to upload {len(frames) - successful_uploads} frames for video {video_id}")
        
        return successful_uploads

    except Exception as e:
        vlogger.logger.error(f"Error processing batch for video {video_id}: {str(e)}", exc_info=True)
        return 0
########################################################

## Get first video frame
########################################################
async def get_first_video_frame(video_id: str):
    app_logger.log_info(f"Received request for first frame of video: {video_id}")
    
    # Construct the path to the first frame
    first_frame_path = f'{video_id}/frames/000000.jpg'
    
    try:
        # Check if the object exists
        await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=first_frame_path
        )
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            app_logger.log_error(f"First frame not found for video: {video_id}")
            raise HTTPException(status_code=404, detail="First frame not found")
        else:
            app_logger.log_error(f"Error checking first frame for video {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error checking first frame")
    
    try:
        # Download the frame data
        app_logger.log_info(f"Downloading first frame for video: {video_id}")
        response = await asyncio.to_thread(
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=first_frame_path
        )
        frame_data = response['Body'].read()
        
        app_logger.log_info(f"Successfully retrieved first frame for video: {video_id}")
        return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
    
    except Exception as e:
        app_logger.log_error(f"Error retrieving first frame for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving first frame")

# async def get_video_frame(video_id: str):
#     logger.info(f"Received request for first non-black frame of video: {video_id}")
#     bucket = storage_client.bucket(PROCESSING_BUCKET)
    
#     frames_prefix = f'{video_id}/frames/'
#     blobs = list(bucket.list_blobs(prefix=frames_prefix))
#     blobs.sort(key=lambda x: int(re.findall(r'\d+', x.name)[-1]))
    
#     frame_index = 0
#     while frame_index < len(blobs):
#         blob = blobs[frame_index]
#         frame_data = blob.download_as_bytes()
#         nparr = np.frombuffer(frame_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
#         # Check if the frame is not completely black
#         if np.mean(img) > 120:  # You can adjust this threshold as needed
#             logger.info(f"Found non-black frame: {blob.name}")
#             return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
        
#         # If black, jump forward 30 frames or to the end
#         frame_index += min(30, len(blobs) - frame_index - 1)
    
#     logger.warning(f"No non-black frame found for video: {video_id}")
#     raise HTTPException(status_code=404, detail="No non-black frame found")
########################################################

## Get all video frames
########################################################
async def get_all_video_frames(video_id: str) -> List[dict]:
    app_logger.log_info(f"Received request for video frames: {video_id}")
    frames_prefix = f'{video_id}/frames/'
    
    try:
        # List objects with the frames prefix
        app_logger.log_info(f"Listing objects with prefix: {frames_prefix}")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=settings.PROCESSING_BUCKET, Prefix=frames_prefix)
        
        frames = []
        total_objects = 0
        processed_objects = 0
        
        for page in pages:
            total_objects += len(page.get('Contents', []))
            for obj in page.get('Contents', []):
                try:
                    frame_number = int(obj['Key'].split('/')[-1].split('.')[0])
                    if frame_number % 50 == 0:
                        # Generate a pre-signed URL for the frame
                        signed_url = await asyncio.to_thread (
                            s3_client.generate_presigned_url,
                            'get_object',
                            Params={'Bucket': settings.PROCESSING_BUCKET, 'Key': obj['Key']},
                            ExpiresIn=3600
                        )
                        frames.append({
                            "number": frame_number,
                            "url": signed_url
                        })
                    processed_objects += 1
                except Exception as e:
                    app_logger.log_error(f"Error generating signed URL for object {obj['Key']}: {str(e)}", exc_info=True)
            
            app_logger.log_info(f"Processed {processed_objects}/{total_objects} objects")

        sorted_frames = sorted(frames, key=lambda x: x["number"])
        app_logger.log_info(f"Returning {len(sorted_frames)} frames for video {video_id}")
        return sorted_frames

    except Exception as e:
        app_logger.log_error(f"Error processing frames for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")
########################################################