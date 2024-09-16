import cv2
import time
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List
from io import BytesIO
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from services import s3_operations

## Process video frames
########################################################
async def process_video_frames(video_path: str, video_id: str, s3_client, status_tracker: StatusTracker):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frame_number = 0
    batches = []
    current_batch = []

    status_tracker.update_process_status("video_processing", "in_progress", 0)

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
        status_tracker.update_process_status("video_processing", "in_progress", progress)

    cap.release()

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
        frame_futures = [executor.submit(process_batch, batch, video_id, s3_client) for batch in batches]
        await asyncio.get_event_loop().run_in_executor(None, concurrent.futures.wait, frame_futures)

    processing_time = time.time() - start_time
    extracted_frames = frame_number // settings.FRAME_INTERVAL
    processing_fps = extracted_frames / processing_time
    processing_speed = (processing_fps / fps) * 100

    status_tracker.update_process_status("video_processing", "complete", 100)

    return {
        "video_length": f'{duration:.1f} seconds',
        "total_frames": frame_count,
        "extracted_frames": extracted_frames,
        "video_fps": round(fps, 2),
        "video_processing_time": processing_time,
        "video_processing_fps": processing_fps,
        "video_processing_speed": processing_speed
    }
########################################################

## Process batches of frames
########################################################
def process_batch(batch, video_id, s3_client):
    try:
        uploads = []
        for frame, frame_number in batch:
            frame_filename = f'{frame_number:06d}.jpg'
            _, buffer = cv2.imencode('.jpg', frame)
            uploads.append((f'{video_id}/frames/{frame_filename}', buffer.tobytes()))

        # Perform batch upload
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(uploads), 10)) as executor:
            results = list(executor.map(
                lambda x: s3_operations.upload_frame_to_s3(s3_client, settings.PROCESSING_BUCKET, x[0], x[1]), 
                uploads
            ))
        
        successful_uploads = sum(results)
        # logger.info(f"Successfully uploaded {successful_uploads} out of {len(uploads)} frames for video {video_id}")
        
        if successful_uploads != len(uploads):
            logger.warning(f"Failed to upload {len(uploads) - successful_uploads} frames for video {video_id}")
    
    except Exception as e:
        logger.error(f"Error processing batch for video {video_id}: {str(e)}", exc_info=True)
########################################################

## Get first video frame
########################################################
async def get_first_video_frame(video_id: str):
    # logger.info(f"Received request for first frame of video: {video_id}")
    s3_client = get_s3_client()
    
    # Construct the path to the first frame
    first_frame_path = f'{video_id}/frames/000000.jpg'
    
    try:
        # Check if the object exists
        s3_client.head_object(Bucket=settings.PROCESSING_BUCKET, Key=first_frame_path)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"First frame not found for video: {video_id}")
            raise HTTPException(status_code=404, detail="First frame not found")
        else:
            logger.error(f"Error checking first frame for video {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error checking first frame")

    try:
        # Download the frame data
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=first_frame_path)
        frame_data = response['Body'].read()
        
        # logger.info(f"Successfully retrieved first frame for video: {video_id}")
        return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
    
    except Exception as e:
        logger.error(f"Error retrieving first frame for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving first frame")

# @app.get("/{video_id}/images/first-frame")
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
    # logger.info(f"Received request for video frames: {video_id}")
    frames_prefix = f'{video_id}/frames/'
    s3_client = get_s3_client()
    
    try:
        # List objects with the frames prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=settings.PROCESSING_BUCKET, Prefix=frames_prefix)
        
        frames = []
        for page in pages:
            for obj in page.get('Contents', []):
                try:
                    frame_number = int(obj['Key'].split('/')[-1].split('.')[0])
                    if frame_number % 50 == 0:
                        # Generate a pre-signed URL for the frame
                        signed_url = s3_client.generate_presigned_url('get_object',
                                                                      Params={'Bucket': settings.PROCESSING_BUCKET,
                                                                              'Key': obj['Key']},
                                                                      ExpiresIn=3600)
                        frames.append({
                            "number": frame_number,
                            "url": signed_url
                        })
                except Exception as e:
                    logger.error(f"Error generating signed URL for object {obj['Key']}: {str(e)}", exc_info=True)
        
        # logger.info(f"Returning {len(frames)} frames for video {video_id}")
        return sorted(frames, key=lambda x: x["number"])
    
    except Exception as e:
        logger.error(f"Error processing frames for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")
########################################################