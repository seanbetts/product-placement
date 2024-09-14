import cv2
import csv
import os
import json
import tempfile
import uuid
import datetime
import time
import asyncio
import subprocess
import concurrent.futures
import io
import logging
import numpy as np
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from processing.ocr_processing import process_ocr, post_process_ocr

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

#Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI server...")

# Set up FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '30'))
FRAME_INTERVAL = int(os.getenv('FRAME_INTERVAL', '1'))
STATUS_UPDATE_INTERVAL = int(os.getenv('STATUS_UPDATE_INTERVAL', '3'))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')

# Configure retry strategy
retry_config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'adaptive'
    }
)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
    config=retry_config
)

class StatusTracker:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.start_time = time.time()
        self.status: Dict[str, Any] = {
            "video_id": video_id,
            "status": "processing",
            "progress": 0,
            "video_processing": {"status": "pending", "progress": 0},
            "audio_extraction": {"status": "pending", "progress": 0},
            "transcription": {"status": "pending", "progress": 0},
            "ocr": {"status": "pending", "progress": 0}
        }
        self.process_events = {
            "video_processing": asyncio.Event(),
            "audio_extraction": asyncio.Event(),
            "transcription": asyncio.Event(),
            "ocr": asyncio.Event()
        }
        self.status["error"] = None

    def update_process_status(self, process: str, status: str, progress: float):
        self.status[process]["status"] = status
        self.status[process]["progress"] = progress
        if status == "complete":
            self.process_events[process].set()

    def calculate_overall_progress(self):
        total_progress = sum(self.status[process]["progress"] for process in self.process_events)
        self.status["progress"] = total_progress / len(self.process_events)

    async def wait_for_completion(self):
        await asyncio.gather(*[event.wait() for event in self.process_events.values()])
        self.status["status"] = "complete"

    def get_status(self):
        elapsed_time = time.time() - self.start_time
        self.status["elapsed_time"] = f"{elapsed_time:.2f} seconds"
        return self.status

    def update_s3_status(self, s3_client):
        current_status = self.get_status()
        status_key = f'{self.video_id}/status.json'
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=status_key,
            Body=json.dumps(current_status),
            ContentType='application/json'
        )

    def set_error(self, error_message: str):
        self.status["error"] = error_message
        self.status["status"] = "error"
        self.update_s3_status(s3_client)


class NameUpdate(BaseModel):
    name: str

# Dictionary to keep track of active uploads
active_uploads: Dict[str, bool] = {}

########################################################
## FAST API ENDPOINTS                                 ##
########################################################

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
@app.get("/health")
async def health_check():
    logger.debug("Health check called")
    return {"status": "product placement backend ok"}
########################################################

## UPLOAD ENDPOINT (POST)
## Uploads a video to the processing bucket and schedules the video processing
@app.post("/video/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    video_id: Optional[str] = Form(None),
):
    if not video_id:
        video_id = str(uuid.uuid4())
    
    log_context = {
        "video_id": video_id,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks,
        "upload_filename": file.filename,
        "content_type": file.content_type,
    }
    logger.info("Received upload request", extra=log_context)

    # Mark this upload as active
    active_uploads[video_id] = True

    s3_key = f'{video_id}/original.mp4'

    try:
        chunk = await file.read()
        
        # Create a temporary file to store the video chunks
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            
            if chunk_number > 1:
                # If not the first chunk, download existing content
                s3_client.download_file(PROCESSING_BUCKET, s3_key, temp_filename)
            
            # Append the new chunk
            with open(temp_filename, 'ab') as f:
                f.write(chunk)
            
            # Check if upload has been cancelled
            if not active_uploads.get(video_id, False):
                raise Exception("Upload cancelled")

            # Upload the combined content back to S3
            s3_client.upload_file(temp_filename, PROCESSING_BUCKET, s3_key)
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        logger.info(f"Chunk {chunk_number}/{total_chunks} uploaded successfully", extra=log_context)
        
        if chunk_number == total_chunks:
            logger.info("File upload completed successfully", extra=log_context)
            del active_uploads[video_id]  # Remove from active uploads
            background_tasks.add_task(run_video_processing, video_id)
            return {"video_id": video_id, "status": "processing"}
        else:
            return {
                "video_id": video_id,
                "status": "uploading",
                "chunk": chunk_number,
            }

    except Exception as e:
        logger.error("Error during chunk upload", exc_info=True, extra={**log_context, "error": str(e)})
        # Clean up if there was an error
        if video_id in active_uploads:
            del active_uploads[video_id]
        try:
            s3_client.delete_object(Bucket=PROCESSING_BUCKET, Key=s3_key)
        except:
            pass
        return JSONResponse(status_code=500, content={"error": "Upload failed", "video_id": video_id, "chunk": chunk_number, "details": str(e)})
########################################################

## CANCEL UPLOAD ENDPOINT (POST)
## Cancels an upload for a given video ID
@app.post("/video/cancel-upload/{video_id}")
async def cancel_upload(video_id: str):
    if video_id in active_uploads:
        active_uploads[video_id] = False
        s3_key = f'{video_id}/original.mp4'
        try:
            # Check if the object exists
            s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=s3_key)
            # If it exists, delete it
            s3_client.delete_object(Bucket=PROCESSING_BUCKET, Key=s3_key)
            logger.info(f"Upload cancelled for video_id: {video_id}")
            return {"status": "cancelled", "video_id": video_id}
        except s3_client.exceptions.ClientError as e:
            # If the object was not found, it's okay, just log it
            if e.response['Error']['Code'] == '404':
                logger.info(f"No file found to delete for cancelled upload: {video_id}")
            else:
                logger.error(f"Error deleting file for cancelled upload: {video_id}", exc_info=True)
        return {"status": "cancelled", "video_id": video_id}
    else:
        return JSONResponse(status_code=404, content={"error": "Upload not found", "video_id": video_id})
########################################################

## PROCESSED VIDEOS ENDPOINT (GET)
## Returns a list of all processed videos
@app.get("/video/processed-videos")
async def get_processed_videos():
    logger.info("Received request for processed videos")
    completed_videos_key = '_completed_videos.json'
    
    try:
        # Get the list of completed video IDs
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=completed_videos_key)
        completed_video_ids = json.loads(response['Body'].read().decode('utf-8'))
        
        processed_videos = []
        for video_id in completed_video_ids:
            try:
                # Get the processing_stats.json for this video
                stats_key = f'{video_id}/processing_stats.json'
                stats_response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=stats_key)
                stats_data = json.loads(stats_response['Body'].read().decode('utf-8'))
                
                processed_videos.append({
                    'video_id': video_id,
                    'details': stats_data
                })
            except Exception as e:
                logger.error(f"Error retrieving details for video {video_id}: {str(e)}")
        
        logger.info(f"Returning {len(processed_videos)} processed videos")
        return processed_videos
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info("No completed videos found")
            return []
        else:
            logger.error(f"Error retrieving completed videos list: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving completed videos")
########################################################  

## STATUS ENDPOINT (GET)
## Returns status.json for a given video ID 
@app.get("/{video_id}/video/status")
async def get_status(video_id: str):
    logger.info(f"Received status request for video ID: {video_id}")
    
    status_key = f'{video_id}/status.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=status_key)
        status_data = json.loads(response['Body'].read().decode('utf-8'))
        
        logger.info(f"Status for video ID {video_id}: {status_data}")
        return status_data
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"Status not found for video ID: {video_id}")
            raise HTTPException(status_code=404, detail="Video status not found")
        else:
            logger.error(f"Error retrieving status for video ID {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving video status")
########################################################  

## PROCESSING STATS ENDPOINT (GET)
## Returns the processing_stats.json for a given video ID
@app.get("/{video_id}/video/processing-stats")
async def get_processing_stats(video_id: str):
    logger.info(f"Received request for processing stats of video: {video_id}")
    
    stats_key = f'{video_id}/processing_stats.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=stats_key)
        stats = json.loads(response['Body'].read().decode('utf-8'))
        return stats
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Processing stats not found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Processing stats not found")
    except Exception as e:
        logger.error(f"Error retrieving processing stats for video ID {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processing stats")
########################################################

## UPDATE VIDEO NAME ENDPOINT (POST)
## Updates the name of a video
@app.post("/{video_id}/video/update-name")
async def update_video_name(video_id: str, name_update: NameUpdate):
    logger.info(f"Received request to update name for video {video_id} to '{name_update.name}'")
    
    status_key = f'{video_id}/status.json'
    stats_key = f'{video_id}/processing_stats.json'

    try:
        # Check if both files exist
        s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=status_key)
        s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=stats_key)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise HTTPException(status_code=404, detail="Video not found")
        else:
            raise HTTPException(status_code=500, detail="Error checking video files")

    if not name_update.name or name_update.name.strip() == "":
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    try:
        # Update status.json
        status_response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=status_key)
        status_data = json.loads(status_response['Body'].read().decode('utf-8'))
        status_data['name'] = name_update.name
        s3_client.put_object(Bucket=PROCESSING_BUCKET, Key=status_key, 
                             Body=json.dumps(status_data), ContentType='application/json')

        # Update processing_stats.json
        stats_response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=stats_key)
        stats_data = json.loads(stats_response['Body'].read().decode('utf-8'))
        stats_data['name'] = name_update.name
        s3_client.put_object(Bucket=PROCESSING_BUCKET, Key=stats_key, 
                             Body=json.dumps(stats_data), ContentType='application/json')

        return {"message": f"Video name updated successfully to {name_update.name}"}

    except Exception as e:
        logger.error(f"Error updating name for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating video name")
########################################################

## FIRST VIDEO FRAME ENDPOINT (GET)   
## Returns the first frame of the video as a JPEG image
########################################################
@app.get("/{video_id}/images/first-frame")
async def get_video_frame(video_id: str):
    logger.info(f"Received request for first frame of video: {video_id}")
    
    # Construct the path to the first frame
    first_frame_path = f'{video_id}/frames/000000.jpg'
    
    try:
        # Check if the object exists
        s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=first_frame_path)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"First frame not found for video: {video_id}")
            raise HTTPException(status_code=404, detail="First frame not found")
        else:
            logger.error(f"Error checking first frame for video {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error checking first frame")

    try:
        # Download the frame data
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=first_frame_path)
        frame_data = response['Body'].read()
        
        logger.info(f"Successfully retrieved first frame for video: {video_id}")
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

## ALL VIDEO FRAMES ENDPOINT (GET)
## Returns a list of all video frames
@app.get("/{video_id}/images/all-frames")
async def get_video_frames(video_id: str) -> List[dict]:
    logger.info(f"Received request for video frames: {video_id}")
    frames_prefix = f'{video_id}/frames/'
    
    try:
        # List objects with the frames prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=PROCESSING_BUCKET, Prefix=frames_prefix)
        
        frames = []
        for page in pages:
            for obj in page.get('Contents', []):
                try:
                    frame_number = int(obj['Key'].split('/')[-1].split('.')[0])
                    if frame_number % 50 == 0:
                        # Generate a pre-signed URL for the frame
                        signed_url = s3_client.generate_presigned_url('get_object',
                                                                      Params={'Bucket': PROCESSING_BUCKET,
                                                                              'Key': obj['Key']},
                                                                      ExpiresIn=3600)
                        frames.append({
                            "number": frame_number,
                            "url": signed_url
                        })
                except Exception as e:
                    logger.error(f"Error generating signed URL for object {obj['Key']}: {str(e)}", exc_info=True)
        
        logger.info(f"Returning {len(frames)} frames for video {video_id}")
        return sorted(frames, key=lambda x: x["number"])
    
    except Exception as e:
        logger.error(f"Error processing frames for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")
########################################################

## TRANSCRIPT ENDPOINT (GET)
## Returns the transcript.json for a given video ID
@app.get("/{video_id}/transcript")
async def get_transcript(video_id: str):
    transcript_key = f'{video_id}/transcripts/transcript.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=transcript_key)
        transcript = json.loads(response['Body'].read().decode('utf-8'))
        return transcript
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Transcript not found")
    except Exception as e:
        logger.error(f"Error retrieving transcript for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving transcript")
########################################################

## DOWNLOAD ENDPOINT (GET)
## Downloads a file for a given video ID and file type
@app.get("/{video_id}/files/download/{file_type}")
async def download_file(video_id: str, file_type: str):
    logger.info(f"Received download request for {file_type} of video: {video_id}")
    
    if file_type == "video":
        key = f'{video_id}/original.mp4'
        filename = f"{video_id}_video.mp4"
    elif file_type == "audio":
        key = f'{video_id}/audio.mp3'
        filename = f"{video_id}_audio.mp3"
    elif file_type == "transcript":
        key = f'{video_id}/transcripts/transcript.txt'
        filename = f"{video_id}_transcript.txt"
    elif file_type == "word-cloud":
        key = f'{video_id}/ocr/wordcloud.jpg'
        filename = f"{video_id}_wordcloud.jpg"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Check if the object exists
        s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=key)
        
        # Create a pre-signed URL for the object
        url = s3_client.generate_presigned_url('get_object',
                                               Params={'Bucket': PROCESSING_BUCKET, 'Key': key},
                                               ExpiresIn=3600,
                                               HttpMethod='GET')
        
        # Redirect to the pre-signed URL
        return RedirectResponse(url=url)
    
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} not found")
        else:
            logger.error(f"Error downloading {file_type} for video {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading {file_type}")
########################################################

## WORD CLOUD ENDPOINT (GET)
## Returns the wordcloud.jpg for a given video ID
@app.get("/{video_id}/ocr/wordcloud")
async def get_word_cloud(video_id: str):
    logger.info(f"Received request for word cloud of video: {video_id}")
    wordcloud_key = f'{video_id}/ocr/wordcloud.jpg'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=wordcloud_key)
        image_data = response['Body'].read()
        return StreamingResponse(BytesIO(image_data), media_type="image/jpeg")
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Text Detection word cloud not found")
    except Exception as e:
        logger.error(f"Error retrieving word cloud for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## BRANDS OCR TABLE ENDPOINT (GET)
## Returns the brands_table.json for a given video ID
@app.get("/{video_id}/ocr/brands-ocr-table")
async def get_processed_ocr_results(video_id: str):
    logger.info(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_table.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Brands OCR results not found")
    except Exception as e:
        logger.error(f"Error retrieving brand OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR results")
########################################################

## REPROCESS OCR ENDPOINT (POST)
## Reprocesses the OCR for a given video ID
@app.post("/{video_id}/ocr/reprocess-ocr")
async def reprocess_ocr(video_id: str):
    logger.info(f"Received request to reprocess OCR for video: {video_id}")
    stats_key = f'{video_id}/processing_stats.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=stats_key)
        stats = json.loads(response['Body'].read().decode('utf-8'))
        fps = float(stats['video']['video_fps'])
        video_resolution = await get_video_resolution(video_id)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Processing stats not found")
    except Exception as e:
        logger.error(f"Error retrieving processing stats for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processing stats")

    try:
        processed_results = await post_process_ocr(video_id, fps, video_resolution, s3_client)
        if processed_results:
            return {"status": "success", "message": "OCR results reprocessed and saved"}
        else:
            raise HTTPException(status_code=404, detail="OCR results not found for reprocessing")
    except Exception as e:
        logger.error(f"Error reprocessing OCR for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reprocessing OCR")
########################################################

## OCR RESULTS ENDPOINT (GET)
## Returns the ocr_results.json for a given video ID
@app.get("/{video_id}/ocr/results")
async def get_ocr_results(video_id: str):
    logger.info(f"Received request for OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/ocr_results.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="OCR results not found")
    except Exception as e:
        logger.error(f"Error retrieving OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving OCR results")
########################################################

## PROCESSED OCR RESULTS ENDPOINT (GET)
## Returns the processed_ocr.json for a given video ID
@app.get("/{video_id}/ocr/processed-ocr")
async def get_processed_ocr_results(video_id: str):
    logger.info(f"Received request for processed OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/processed_ocr.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Processed OCR results not found")
    except Exception as e:
        logger.error(f"Error retrieving processed OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results")
########################################################

## BRANDS OCR ENDPOINT (GET)
## Returns the brands_ocr.json for a given video ID
@app.get("/{video_id}/ocr/brands-ocr")
async def get_brands_ocr_results(video_id: str):
    logger.info(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_ocr.json'
    
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Brands OCR results not found")
    except Exception as e:
        logger.error(f"Error retrieving brand OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR results")
########################################################
    
########################################################
## FUNCTIONS                                          ##
########################################################
def update_completed_videos_list(video_id: str):
    completed_videos_key = '_completed_videos.json'
    
    try:
        # Try to get the existing list
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=completed_videos_key)
        completed_videos = json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # If the file doesn't exist, start with an empty list
            completed_videos = []
        else:
            raise

    # Add the new video ID if it's not already in the list
    if video_id not in completed_videos:
        completed_videos.append(video_id)

    # Upload the updated list back to S3
    s3_client.put_object(
        Bucket=PROCESSING_BUCKET,
        Key=completed_videos_key,
        Body=json.dumps(completed_videos),
        ContentType='application/json'
    )

    logger.info(f"Added video {video_id} to completed videos list")

def mark_video_as_completed(video_id: str):
    # Update the processing_stats.json to mark it as completed
    stats_key = f'{video_id}/processing_stats.json'
    try:
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=stats_key)
        stats_data = json.loads(response['Body'].read().decode('utf-8'))
        stats_data['status'] = 'completed'
        stats_data['completion_time'] = datetime.datetime.utcnow().isoformat()
        
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=stats_key,
            Body=json.dumps(stats_data),
            ContentType='application/json'
        )
        
        # Update the completed videos list
        update_completed_videos_list(video_id)
        
        logger.info(f"Marked video {video_id} as completed")
    except Exception as e:
        logger.error(f"Error marking video {video_id} as completed: {str(e)}")

async def run_video_processing(video_id: str):
    try:
        await process_video(video_id)
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)

async def process_video(video_id: str):
    logger.info(f"Starting to process video: {video_id}")
    video_key = f'{video_id}/original.mp4'

    status_tracker = StatusTracker(video_id)
    status_tracker.update_s3_status(s3_client)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        s3_client.download_file(PROCESSING_BUCKET, video_key, temp_video.name)
        temp_video_path = temp_video.name

    try:
        total_start_time = time.time()
        
        # Start status update task
        status_update_task = asyncio.create_task(periodic_status_update(video_id, status_tracker, s3_client))

        # Run video frame processing and audio extraction in parallel
        video_task = asyncio.create_task(process_video_frames(temp_video_path, video_id, s3_client, status_tracker))
        audio_task = asyncio.create_task(extract_audio(temp_video_path, video_id, s3_client, status_tracker))

        # Wait for both tasks to complete
        video_stats, audio_stats = await asyncio.gather(video_task, audio_task)

        if status_tracker.status.get("error"):
            logger.error(f"Error encountered during video/audio processing: {status_tracker.status['error']}")
            return

        # Start transcription only after audio extraction is complete
        # transcription_start_time = time.time()
        transcription_stats = await transcribe_audio(video_id, s3_client, float(video_stats['video_length'].split()[0]), status_tracker)
        
        if status_tracker.status.get("error"):
            logger.error(f"Error encountered during transcription: {status_tracker.status['error']}")
            return

        # transcription_processing_time = time.time() - transcription_start_time

        # Start OCR processing
        ocr_start_time = time.time()
        ocr_stats = await process_ocr(video_id, status_tracker, s3_client)

        if status_tracker.status.get("error"):
            logger.error(f"Error encountered during OCR processing: {status_tracker.status['error']}")
            return

        # Post-process OCR results
        video_resolution = await get_video_resolution(video_id)
        brand_results = await post_process_ocr(video_id, video_stats['video_fps'], video_resolution, s3_client)

        # ocr_processing_time = time.time() - ocr_start_time

        # Wait for all processes to complete
        await status_tracker.wait_for_completion()

        # Cancel the status update task
        status_update_task.cancel()

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        processing_stats = {
            "video_id": video_id,
            "video_length": video_stats['video_length'],
            "video": {
                "total_frames": video_stats['total_frames'],
                "extracted_frames": video_stats['extracted_frames'],
                "video_fps": video_stats['video_fps'],
                "video_processing_time": f"{video_stats['video_processing_time']:.2f} seconds",
                "video_processing_fps": f"{video_stats['video_processing_fps']:.2f}",
                "video_processing_speed": f"{video_stats['video_processing_speed']:.1f}% of real-time",
            },
            "audio": {
                "audio_length": audio_stats['audio_length'],
                "audio_processing_time": f"{audio_stats['audio_processing_time']:.2f} seconds",
                "audio_processing_speed": f"{audio_stats['audio_processing_speed']:.1f}% of real-time",
            },
            "transcription": {
                "transcription_processing_time": f"{transcription_stats['transcription_time']:.2f} seconds",
                "word_count": transcription_stats['word_count'],
                "confidence": f"{transcription_stats['overall_confidence']:.1f}%",
                "transcription_speed": f"{transcription_stats['transcription_speed']:.1f}% of real-time"
            },
            "ocr": {
                "ocr_processing_time": ocr_stats['ocr_processing_time'],
                "frames_processed": ocr_stats['frames_processed'],
                "frames_with_text": ocr_stats['frames_with_text'],
                "total_words_detected": ocr_stats['total_words_detected']
            },
            "total_processing_start_time": datetime.datetime.fromtimestamp(total_start_time).isoformat(),
            "total_processing_end_time": datetime.datetime.fromtimestamp(total_end_time).isoformat(),
            "total_processing_time": f"{total_processing_time:.2f} seconds",
            "total_processing_speed": f"{(float(video_stats['video_length'].split()[0]) / total_processing_time * 100):.1f}% of real-time"
        }

        # Save processing stats to a new file
        stats_key = f'{video_id}/processing_stats.json'
        s3_client.put_object(Bucket=PROCESSING_BUCKET, Key=stats_key, 
                             Body=json.dumps(processing_stats, indent=2), 
                             ContentType='application/json')

        # Update final status
        status_tracker.status["status"] = "complete"
        status_tracker.update_s3_status(s3_client)

        # Mark video as completed
        mark_video_as_completed(video_id)

        logger.info(f"Completed processing video: {video_id}")
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        status_tracker.set_error(str(e))
        status_tracker.update_s3_status(s3_client)
    finally:
        os.unlink(temp_video_path)

async def periodic_status_update(video_id: str, status_tracker: StatusTracker, s3_client):
    while True:
        status_tracker.calculate_overall_progress()
        status_tracker.update_s3_status(s3_client)
        await asyncio.sleep(STATUS_UPDATE_INTERVAL)

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

        if frame_number % FRAME_INTERVAL == 0:
            current_batch.append((frame, frame_number))

            if len(current_batch) == BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []

        frame_number += 1

        progress = (frame_number / frame_count) * 100
        status_tracker.update_process_status("video_processing", "in_progress", progress)

    cap.release()

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        frame_futures = [executor.submit(process_batch, batch, video_id, s3_client) for batch in batches]
        await asyncio.get_event_loop().run_in_executor(None, concurrent.futures.wait, frame_futures)

    processing_time = time.time() - start_time
    extracted_frames = frame_number // FRAME_INTERVAL
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

async def extract_audio(video_path: str, video_id: str, s3_client, status_tracker: StatusTracker):
    logger.info(f"Extracting audio for video: {video_id}")
    audio_path = f"/tmp/{video_id}_audio.mp3"
    start_time = time.time()
    status_tracker.update_process_status("audio_extraction", "in_progress", 0)

    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            audio_path
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stderr.decode())

        # Upload audio file to S3
        with open(audio_path, 'rb') as audio_file:
            s3_client.put_object(
                Bucket=PROCESSING_BUCKET,
                Key=f'{video_id}/audio.mp3',
                Body=audio_file,
                ContentType='audio/mpeg'
            )
        logger.info(f"Audio extracted and uploaded for video: {video_id}")

        # Extract audio duration using ffprobe
        duration_command = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            audio_path
        ]
        duration_process = await asyncio.create_subprocess_exec(
            *duration_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await duration_process.communicate()
        audio_duration = float(stdout.decode().strip())

        processing_time = time.time() - start_time
        processing_speed = (audio_duration / processing_time) * 100

        status_tracker.update_process_status("audio_extraction", "complete", 100)

        return {
            "audio_length": f"{audio_duration:.2f} seconds",
            "audio_processing_time": processing_time,
            "audio_processing_speed": processing_speed
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting audio for video {video_id}: {str(e)}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        status_tracker.update_process_status("audio_extraction", "error", 0)
    except Exception as e:
        logger.error(f"Unexpected error extracting audio for video {video_id}: {str(e)}", exc_info=True)
        status_tracker.update_process_status("audio_extraction", "error", 0)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {
        "audio_length": "Unknown",
        "audio_processing_time": time.time() - start_time,
        "audio_processing_speed": 0
    }

async def transcribe_audio(video_id: str, s3_client, video_length: float, status_tracker: StatusTracker):
    logger.info(f"Transcribing audio for video: {video_id}")
    audio_key = f'{video_id}/audio.mp3'
    transcript_key = f"{video_id}/transcripts/audio_transcript_{video_id}.json"

    status_tracker.update_process_status("transcription", "in_progress", 0)

    try:
        # Check if the audio file exists
        s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=audio_key)
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"Audio file not found for video: {video_id}")
            status_tracker.update_process_status("transcription", "error", 0)
            return None
        else:
            raise

    try:
        # Instantiate the Transcribe client
        transcribe_client = boto3.client('transcribe')

        # Set up the transcription job
        job_name = f"audio_transcript_{video_id}"
        job_uri = f"s3://{PROCESSING_BUCKET}/{audio_key}"
        
        transcription_start_time = time.time()
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat='mp3',
            LanguageCode='en-US',
            OutputBucketName=PROCESSING_BUCKET,
            OutputKey=f"{video_id}/transcripts/",
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 10,
                'ShowAlternatives': False
            }
        )
        
        logger.info(f"Started transcription job for video {video_id}")

        # Wait for the job to complete with a timeout of 2x video length
        timeout = video_length * 2
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                break
            
            await asyncio.sleep(5)
            elapsed_time = time.time() - transcription_start_time
            progress = min(100, (elapsed_time / timeout) * 100)
            status_tracker.update_process_status("transcription", "in_progress", progress)

        transcription_end_time = time.time()
        
        if job_status == 'COMPLETED':
            logger.info(f"Transcription completed for video {video_id}")
            
            # Wait for the transcript file to be available in S3
            max_retries = 10
            for i in range(max_retries):
                try:
                    s3_client.head_object(Bucket=PROCESSING_BUCKET, Key=transcript_key)
                    logger.info(f"Transcript file found for video {video_id}")
                    break
                except s3_client.exceptions.ClientError:
                    if i < max_retries - 1:
                        await asyncio.sleep(5)
                    else:
                        raise FileNotFoundError(f"Transcript file not found for video {video_id} after {max_retries} retries")

            # Process the response and create transcripts
            plain_transcript, json_transcript, word_count, overall_confidence = await process_transcription_response(s3_client, video_id)

            logger.info(f"Transcripts processed and uploaded for video: {video_id}")

            # Calculate transcription stats
            transcription_time = transcription_end_time - transcription_start_time
            transcription_speed = (video_length / transcription_time) * 100 if transcription_time > 0 else 0
            overall_confidence = overall_confidence * 100

            status_tracker.update_process_status("transcription", "complete", 100)

            return {
                "word_count": word_count,
                "transcription_time": transcription_time,
                "transcription_speed": transcription_speed,
                "overall_confidence": overall_confidence
            }
        else:
            logger.error(f"Transcription job failed for video {video_id}")
            status_tracker.update_process_status("transcription", "error", 0)
            return None

    except Exception as e:
        logger.error(f"Error transcribing audio for video {video_id}: {str(e)}", exc_info=True)
        status_tracker.update_process_status("transcription", "error", 0)
        return None

async def process_transcription_response(s3_client, video_id: str):
    plain_transcript = ""
    json_transcript = []
    word_count = 0
    total_confidence = 0

    try:
        # Find the transcript JSON file
        transcript_key = f"{video_id}/transcripts/audio_transcript_{video_id}.json"
        
        try:
            response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=transcript_key)
            transcript_content = response['Body'].read().decode('utf-8')
            transcript_data = json.loads(transcript_content)
        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"No transcript file found for video {video_id}")

        # Process the transcript data
        items = transcript_data['results']['items']
        for item in items:
            if item['type'] == 'pronunciation':
                word = item['alternatives'][0]['content']
                confidence = float(item['alternatives'][0]['confidence'])
                start_time = item.get('start_time', '0')
                end_time = item.get('end_time', '0')

                word_count += 1
                total_confidence += confidence
                
                json_transcript.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "word": word,
                    "confidence": confidence
                })

                plain_transcript += word + " "
            elif item['type'] == 'punctuation':
                plain_transcript = plain_transcript.rstrip() + item['alternatives'][0]['content'] + " "

        # Clean up the plain transcript
        plain_transcript = plain_transcript.strip()

        # Calculate overall confidence score
        overall_confidence = total_confidence / word_count if word_count > 0 else 0

        # Save transcript.json
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/transcripts/transcript.json',
            Body=json.dumps(json_transcript, indent=2),
            ContentType='application/json'
        )

        # Save transcript.txt
        s3_client.put_object(
            Bucket=PROCESSING_BUCKET,
            Key=f'{video_id}/transcripts/transcript.txt',
            Body=plain_transcript,
            ContentType='text/plain'
        )

    except FileNotFoundError as e:
        logger.error(f"Transcript file not found for video {video_id}: {str(e)}")
        plain_transcript = "Transcript file not found."
        json_transcript = []
        word_count = 0
        overall_confidence = 0
    except Exception as e:
        logger.error(f"Error processing transcription response for video {video_id}: {str(e)}", exc_info=True)
        plain_transcript = "Error processing transcription."
        json_transcript = []
        word_count = 0
        overall_confidence = 0

    return plain_transcript, json_transcript, word_count, overall_confidence

def upload_frame_to_s3(s3_client, bucket, key, body):
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/jpeg')
        return True
    except Exception as e:
        logger.error(f"Failed to upload frame {key}: {str(e)}")
        return False

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
                lambda x: upload_frame_to_s3(s3_client, PROCESSING_BUCKET, x[0], x[1]), 
                uploads
            ))
        
        successful_uploads = sum(results)
        logger.info(f"Successfully uploaded {successful_uploads} out of {len(uploads)} frames for video {video_id}")
        
        if successful_uploads != len(uploads):
            logger.warning(f"Failed to upload {len(uploads) - successful_uploads} frames for video {video_id}")
    
    except Exception as e:
        logger.error(f"Error processing batch for video {video_id}: {str(e)}", exc_info=True)

async def get_video_resolution(video_id: str) -> Tuple[int, int]:
    try:
        # Construct the path to the first frame
        first_frame_path = f'{video_id}/frames/000000.jpg'
        
        # Download the frame data
        response = s3_client.get_object(Bucket=PROCESSING_BUCKET, Key=first_frame_path)
        frame_data = response['Body'].read()
        
        # Open the image using PIL
        with Image.open(io.BytesIO(frame_data)) as img:
            width, height = img.size
        
        logger.info(f"Detected resolution for video {video_id}: {width}x{height}")
        return (width, height)
    
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"First frame not found for video {video_id}")
        raise FileNotFoundError(f"First frame not found for video {video_id}")
    except Exception as e:
        logger.error(f"Unexpected error retrieving video resolution for {video_id}: {str(e)}")
        # Return a default resolution if unable to retrieve
        logger.warning(f"Using default resolution (1920x1080) for video {video_id}")
        return (1920, 1080)  # Default to 1080p

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Environment variables: {os.environ}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")