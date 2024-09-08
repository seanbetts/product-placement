import cv2
import os
import json
import tempfile
import uuid
import logging
import datetime
import time
import asyncio
import subprocess
import urllib3
import concurrent.futures
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from google.resumable_media import DataCorruption
from google.auth.transport.requests import AuthorizedSession
from google.api_core import retry
from google.cloud import vision
from google.oauth2 import service_account
from google.auth import default
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pydantic import BaseModel
from processing.ocr_processing import process_ocr, post_process_ocr

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '20'))
FRAME_INTERVAL = int(os.getenv('FRAME_INTERVAL', '1'))
STATUS_UPDATE_INTERVAL = int(os.getenv('STATUS_UPDATE_INTERVAL', '3'))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))

# Custom transport with larger connection pool
class CustomTransport(AuthorizedSession):
    def __init__(self, credentials):
        super().__init__(credentials)
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"]
        )
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry_strategy)
        self.mount("https://", adapter)

# Set up Google Cloud Storage client with proper authentication and custom transport
try:
    if GOOGLE_APPLICATION_CREDENTIALS:
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    else:
        # Use default credentials (this will work in GCP)
        credentials, _ = default()
    
    custom_transport = CustomTransport(credentials)
    storage_client = storage.Client(credentials=credentials, _http=custom_transport)
except Exception as e:
    logger.error(f"Error setting up Google Cloud Storage client: {str(e)}")
    raise

# Set up Google Cloud Vision client
vision_client = vision.ImageAnnotatorClient()

# Increase the connection pool size
urllib3.PoolManager(num_pools=50, maxsize=50)

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

    def update_gcs_status(self, bucket: storage.Bucket):
        status_blob = bucket.blob(f'{self.video_id}/status.json')
        current_status = self.get_status()
        status_blob.upload_from_string(json.dumps(current_status), content_type='application/json')

class NameUpdate(BaseModel):
    name: str

@retry.Retry(predicate=retry.if_exception_type(DataCorruption))
def resumable_upload_with_retry(resumable_upload, session, stream):
    return resumable_upload.transmit_next_chunk(session, stream)

# Dictionary to keep track of active uploads
active_uploads: Dict[str, bool] = {}

########################################################
## FAST API ENDPOINTS                                 ##
########################################################

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "ok"}
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

    bucket = storage_client.bucket(PROCESSING_BUCKET)
    blob = bucket.blob(f'{video_id}/original.mp4')

    try:
        chunk = await file.read()
        
        # Create a temporary file to store the video chunks
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            
            if chunk_number > 1:
                # If not the first chunk, download existing content
                blob.download_to_filename(temp_filename)
            
            # Append the new chunk
            with open(temp_filename, 'ab') as f:
                f.write(chunk)
            
            # Check if upload has been cancelled
            if not active_uploads.get(video_id, False):
                raise Exception("Upload cancelled")

            # Upload the combined content back to GCS
            blob.upload_from_filename(temp_filename)
        
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
        if blob.exists():
            blob.delete()
        return JSONResponse(status_code=500, content={"error": "Upload failed", "video_id": video_id, "chunk": chunk_number, "details": str(e)})
########################################################

## CANCEL UPLOAD ENDPOINT (POST)
## Cancels an upload for a given video ID
@app.post("/video/cancel-upload/{video_id}")
async def cancel_upload(video_id: str):
    if video_id in active_uploads:
        active_uploads[video_id] = False
        bucket = storage_client.bucket(PROCESSING_BUCKET)
        blob = bucket.blob(f'{video_id}/original.mp4')
        if blob.exists():
            blob.delete()
        logger.info(f"Upload cancelled for video_id: {video_id}")
        return {"status": "cancelled", "video_id": video_id}
    else:
        return JSONResponse(status_code=404, content={"error": "Upload not found", "video_id": video_id})
########################################################

## PROCESSED VIDEOS ENDPOINT (GET)
## Returns a list of all processed videos
@app.get("/video/processed-videos")
async def get_processed_videos():
    logger.info("Received request for processed videos")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    processed_videos = []
    
    # List all blobs in the bucket without a delimiter
    blobs = list(bucket.list_blobs())
    logger.info(f"Found {len(blobs)} blobs in the bucket")
    
    for blob in blobs:
        # Identify potential video folders by the presence of a status.json file
        if blob.name.endswith('status.json'):
            video_id = blob.name.rsplit('/', 2)[-2]
            logger.info(f"Found status.json for video: {video_id}")
            
            status_blob = bucket.blob(blob.name)
            stats_blob = bucket.blob(f'{video_id}/processing_stats.json')
            
            if status_blob.exists() and stats_blob.exists():
                try:
                    status_data = json.loads(status_blob.download_as_string())
                    stats_data = json.loads(stats_blob.download_as_string())
                    
                    if status_data.get('status') == 'complete':
                        logger.info(f"Video {video_id} is complete, adding to processed videos")
                        video_data = {
                            'video_id': video_id,
                            'details': stats_data
                        }
                        processed_videos.append(video_data)
                    else:
                        logger.info(f"Video {video_id} is not complete, status: {status_data.get('status')}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON for video {video_id}")
            else:
                logger.info(f"Missing status or stats file for video: {video_id}")
    
    logger.info(f"Returning {len(processed_videos)} processed videos")
    return processed_videos
########################################################  

## STATUS ENDPOINT (GET)
## Returns status.json for a given video ID 
@app.get("/{video_id}/video/status")
async def get_status(video_id: str):
    logger.info(f"Received status request for video ID: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    status_blob = bucket.blob(f'{video_id}/status.json')
    
    if status_blob.exists():
        status_data = json.loads(status_blob.download_as_string())
        logger.info(f"Status for video ID {video_id}: {status_data}")
        return status_data
    else:
        logger.warning(f"Status not found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Video status not found")
########################################################  

## PROCESSING STATS ENDPOINT (GET)
## Returns the processing_stats.json for a given video ID
@app.get("/{video_id}/video/processing-stats")
async def get_processing_stats(video_id: str):
    logger.info(f"Received request for processing stats of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    stats_blob = bucket.blob(f'{video_id}/processing_stats.json')

    if stats_blob.exists():
        stats = json.loads(stats_blob.download_as_string())
        return stats
    else:
        raise HTTPException(status_code=404, detail="Processing stats not found")
########################################################

## UPDATE VIDEO NAME ENDPOINT (POST)
## Updates the name of a video
@app.post("/{video_id}/video/update-name")
async def update_video_name(video_id: str, name_update: NameUpdate):
    logger.info(f"Received request to update name for video {video_id} to '{name_update.name}'")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    status_blob = bucket.blob(f'{video_id}/status.json')
    stats_blob = bucket.blob(f'{video_id}/processing_stats.json')
    
    if not status_blob.exists() or not stats_blob.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    if not name_update.name or name_update.name.strip() == "":
        raise HTTPException(status_code=422, detail="Name cannot be empty")

    # Update status.json
    status_data = json.loads(status_blob.download_as_string())
    status_data['name'] = name_update.name
    status_blob.upload_from_string(json.dumps(status_data), content_type='application/json')

    # Update processing_stats.json
    stats_data = json.loads(stats_blob.download_as_string())
    stats_data['name'] = name_update.name
    stats_blob.upload_from_string(json.dumps(stats_data), content_type='application/json')

    return {f"message": "Video name updated successfully to {name_update.name}"}
########################################################

## VIDEO FRAME ENDPOINT (GET)   
## Returns the first frame of the video as a JPEG image
import cv2
import numpy as np
import re

@app.get("/{video_id}/images/first-frame")
async def get_video_frame(video_id: str):
    logger.info(f"Received request for first non-black frame of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    
    frames_prefix = f'{video_id}/frames/'
    blobs = list(bucket.list_blobs(prefix=frames_prefix))
    blobs.sort(key=lambda x: int(re.findall(r'\d+', x.name)[-1]))
    
    frame_index = 0
    while frame_index < len(blobs):
        blob = blobs[frame_index]
        frame_data = blob.download_as_bytes()
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Check if the frame is not completely black
        if np.mean(img) > 120:  # You can adjust this threshold as needed
            logger.info(f"Found non-black frame: {blob.name}")
            return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
        
        # If black, jump forward 30 frames or to the end
        frame_index += min(30, len(blobs) - frame_index - 1)
    
    logger.warning(f"No non-black frame found for video: {video_id}")
    raise HTTPException(status_code=404, detail="No non-black frame found")
########################################################

## VIDEO FRAMES ENDPOINT (GET)
## Returns a list of all video frames
@app.get("/{video_id}/images/all-frames")
async def get_video_frames(video_id: str) -> List[dict]:
    logger.info(f"Received request for video frames: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    frames_prefix = f'{video_id}/frames/'
    blobs = bucket.list_blobs(prefix=frames_prefix)
    
    frames = []
    for blob in blobs:
        frame_number = int(blob.name.split('/')[-1].split('.')[0])
        if frame_number % 50 == 0:
            frames.append({
                "number": frame_number,
                "url": blob.generate_signed_url(version="v4", expiration=3600, method="GET")
            })
    
    return sorted(frames, key=lambda x: x["number"])
########################################################

## TRANSCRIPT ENDPOINT (GET)
## Returns the transcript.json for a given video ID
@app.get("/{video_id}/transcript")
async def get_transcript(video_id: str):
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    transcript_blob = bucket.blob(f'{video_id}/transcripts/transcript.json')
    
    if transcript_blob.exists():
        transcript = json.loads(transcript_blob.download_as_string())
        return transcript
    else:
        raise HTTPException(status_code=404, detail="Transcript not found")
########################################################

## DOWNLOAD ENDPOINT (GET)
## Downloads a file for a given video ID and file type
@app.get("/{video_id}/files/download/{file_type}")
async def download_file(video_id: str, file_type: str):
    logger.info(f"Received download request for {file_type} of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    
    if file_type == "video":
        blob = bucket.blob(f'{video_id}/original.mp4')
        filename = f"{video_id}_video.mp4"
    elif file_type == "audio":
        blob = bucket.blob(f'{video_id}/audio.mp3')
        filename = f"{video_id}_audio.mp3"
    elif file_type == "transcript":
        blob = bucket.blob(f'{video_id}/transcripts/transcript.txt')
        filename = f"{video_id}_transcript.txt"
    elif file_type == "word-cloud":
        blob = bucket.blob(f'{video_id}/ocr/wordcloud.jpg')
        filename = f"{video_id}_wordcloud.jpg"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if blob.exists():
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        blob.download_to_filename(temp_file.name)
        return FileResponse(temp_file.name, media_type='application/octet-stream', filename=filename)
    else:
        raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} not found")
########################################################

## WORD CLOUD ENDPOINT (GET)
## Returns the wordcloud.jpg for a given video ID
@app.get("/{video_id}/ocr/wordcloud")
async def get_word_cloud(video_id: str):
    logger.info(f"Received request for word cloud of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    wordcloud_blob = bucket.blob(f'{video_id}/ocr/wordcloud.jpg')

    if wordcloud_blob.exists():
        image_data = wordcloud_blob.download_as_bytes()
        return StreamingResponse(BytesIO(image_data), media_type="image/jpeg")
    else:
            raise HTTPException(status_code=404, detail="Text Detection word cloud not found")
########################################################

## BRANDS OCR TABLE ENDPOINT (GET)
## Returns the brands_table.json for a given video ID
@app.get("/{video_id}/ocr/brands-ocr-table")
async def get_processed_ocr_results(video_id: str):
    logger.info(f"Received request for brand OCR results of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    ocr_blob = bucket.blob(f'{video_id}/ocr/brands_table.json')

    if ocr_blob.exists():
        ocr_results = json.loads(ocr_blob.download_as_string())
        return ocr_results
    else:
        raise HTTPException(status_code=404, detail="Brands OCR results not found")
########################################################

## REPROCESS OCR ENDPOINT (POST)
## Reprocesses the OCR for a given video ID
@app.post("/{video_id}/ocr/reprocess-ocr")
async def reprocess_ocr(video_id: str):
    logger.info(f"Received request to reprocess OCR for video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    stats_blob = bucket.blob(f'{video_id}/processing_stats.json')

    if stats_blob.exists():
        stats = json.loads(stats_blob.download_as_string())
        fps = float(stats['video']['video_fps'])
        video_resolution = await get_video_resolution(bucket, video_id)
    else:
        raise HTTPException(status_code=404, detail="Processing stats not found")
    
    processed_results = await post_process_ocr(video_id, fps, video_resolution, bucket)
    if processed_results:
        return {"status": "success", "message": "OCR results reprocessed and saved"}
    else:
        raise HTTPException(status_code=404, detail="OCR results not found for reprocessing")
########################################################

## OCR RESULTS ENDPOINT (GET)
## Returns the ocr_results.json for a given video ID
@app.get("/{video_id}/ocr/results")
async def get_ocr_results(video_id: str):
    logger.info(f"Received request for OCR results of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    ocr_blob = bucket.blob(f'{video_id}/ocr/ocr_results.json')

    if ocr_blob.exists():
        ocr_results = json.loads(ocr_blob.download_as_string())
        return ocr_results
    else:
        raise HTTPException(status_code=404, detail="OCR results not found")
########################################################

## PROCESSED OCR RESULTS ENDPOINT (GET)
## Returns the processed_ocr.json for a given video ID
@app.get("/{video_id}/ocr/processed-ocr")
async def get_processed_ocr_results(video_id: str):
    logger.info(f"Received request for processed OCR results of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    ocr_blob = bucket.blob(f'{video_id}/ocr/processed_ocr.json')

    if ocr_blob.exists():
        ocr_results = json.loads(ocr_blob.download_as_string())
        return ocr_results
    else:
        raise HTTPException(status_code=404, detail="Processed OCR results not found")
########################################################

## BRANDS OCR ENDPOINT (GET)
## Returns the brands_ocr.json for a given video ID
@app.get("/{video_id}/ocr/brands-ocr")
async def get_processed_ocr_results(video_id: str):
    logger.info(f"Received request for brand OCR results of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    ocr_blob = bucket.blob(f'{video_id}/ocr/brands_ocr.json')

    if ocr_blob.exists():
        ocr_results = json.loads(ocr_blob.download_as_string())
        return ocr_results
    else:
        raise HTTPException(status_code=404, detail="Brands OCR results not found")
########################################################
    
########################################################
## FUNCTIONS                                          ##
########################################################

async def run_video_processing(video_id: str):
    try:
        await process_video(video_id)
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)

async def process_video(video_id: str):
    logger.info(f"Starting to process video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    video_blob = bucket.blob(f'{video_id}/original.mp4')

    status_tracker = StatusTracker(video_id)
    status_tracker.update_gcs_status(bucket)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        video_blob.download_to_filename(temp_video.name)
        temp_video_path = temp_video.name

    try:
        total_start_time = time.time()
        
        # Start status update task
        status_update_task = asyncio.create_task(periodic_status_update(video_id, status_tracker, bucket))

        # Run video frame processing and audio extraction in parallel
        video_task = asyncio.create_task(process_video_frames(temp_video_path, video_id, bucket, status_tracker))
        audio_task = asyncio.create_task(extract_audio(temp_video_path, video_id, bucket, status_tracker))

        # Wait for both tasks to complete
        video_stats, audio_stats = await asyncio.gather(video_task, audio_task)

        # Start transcription only after audio extraction is complete
        transcription_start_time = time.time()
        transcription_stats = await transcribe_audio(video_id, bucket, float(video_stats['video_length'].split()[0]), status_tracker)
        transcription_processing_time = time.time() - transcription_start_time

        # Start OCR processing
        ocr_start_time = time.time()
        ocr_stats = await process_ocr(video_id, bucket, status_tracker)

        # Post-process OCR results
        brand_results = await post_process_ocr(video_id, video_stats['video_fps'], bucket)

        ocr_processing_time = time.time() - ocr_start_time

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
        stats_blob = bucket.blob(f'{video_id}/processing_stats.json')
        stats_blob.upload_from_string(json.dumps(processing_stats, indent=2), content_type='application/json')

        # Update final status
        status_tracker.status["status"] = "complete"
        status_tracker.update_gcs_status(bucket)

        logger.info(f"Completed processing video: {video_id}")
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        status_tracker.status["status"] = "error"
        status_tracker.status["error"] = str(e)
        status_tracker.update_gcs_status(bucket)
    finally:
        os.unlink(temp_video_path)

async def periodic_status_update(video_id: str, status_tracker: StatusTracker, bucket: storage.Bucket):
    while True:
        status_tracker.calculate_overall_progress()
        status_tracker.update_gcs_status(bucket)
        await asyncio.sleep(STATUS_UPDATE_INTERVAL)

async def process_video_frames(video_path: str, video_id: str, bucket: storage.Bucket, status_tracker: StatusTracker):
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
        frame_futures = [executor.submit(process_batch, batch, video_id, bucket) for batch in batches]
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

async def extract_audio(video_path: str, video_id: str, bucket: storage.Bucket, status_tracker: StatusTracker):
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

        audio_blob = bucket.blob(f'{video_id}/audio.mp3')
        audio_blob.upload_from_filename(audio_path)
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

async def transcribe_audio(video_id: str, bucket: storage.Bucket, video_length: float, status_tracker: StatusTracker):
    logger.info(f"Transcribing audio for video: {video_id}")
    audio_blob = bucket.blob(f'{video_id}/audio.mp3')

    status_tracker.update_process_status("transcription", "in_progress", 0)

    if not audio_blob.exists():
        logger.warning(f"Audio file not found for video: {video_id}")
        status_tracker.update_process_status("transcription", "error", 0)
        return None

    try:
        # Instantiate the Speech client
        speech_client = SpeechClient()

        # Set up the recognition config
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config={},
            features=cloud_speech.RecognitionFeatures(
                enable_word_confidence=True,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            ),
            model="long",
            language_codes=["en-US"],
        )

        # Set up the audio file metadata
        audio_gcs_uri = f"gs://{PROCESSING_BUCKET}/{video_id}/audio.mp3"
        files = [cloud_speech.BatchRecognizeFileMetadata(uri=audio_gcs_uri)]

        # Set up the output config
        output_config = cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(
                uri=f"gs://{PROCESSING_BUCKET}/{video_id}/transcripts/"
            ),
        )

        # Create the batch recognition request
        request = cloud_speech.BatchRecognizeRequest(
            recognizer=f"projects/{GCP_PROJECT_ID}/locations/global/recognizers/_",
            config=config,
            files=files,
            recognition_output_config=output_config,
        )

        # Start the batch recognition operation
        transcription_start_time = time.time()
        operation = speech_client.batch_recognize(request=request)
        logger.info(f"Started batch recognition for video {video_id}")

        # Wait for the operation to complete with a timeout of 2x video length
        timeout = video_length * 2
        while not operation.done():
            await asyncio.sleep(5)
            elapsed_time = time.time() - transcription_start_time
            progress = min(100, (elapsed_time / timeout) * 100)
            status_tracker.update_process_status("transcription", "in_progress", progress)

        response = operation.result()
        transcription_end_time = time.time()
        logger.info(f"Batch recognition completed for video {video_id}")

        # Process the response and create transcripts
        plain_transcript, json_transcript, word_count, overall_confidence = await process_transcription_response(bucket, video_id)

        # Upload plain transcript
        plain_transcript_blob = bucket.blob(f'{video_id}/transcripts/transcript.txt')
        plain_transcript_blob.upload_from_string(plain_transcript)

        # Upload JSON transcript
        json_transcript_blob = bucket.blob(f'{video_id}/transcripts/transcript.json')
        json_transcript_blob.upload_from_string(json.dumps(json_transcript, indent=2))

        logger.info(f"Transcripts uploaded for video: {video_id}")

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

    except Exception as e:
        logger.error(f"Error transcribing audio for video {video_id}: {str(e)}", exc_info=True)
        status_tracker.update_process_status("transcription", "error", 0)
        return None

async def process_transcription_response(bucket: storage.Bucket, video_id: str):
    plain_transcript = ""
    json_transcript = []
    word_count = 0
    total_confidence = 0

    try:
        # Find the transcript JSON file
        transcript_blobs = list(bucket.list_blobs(prefix=f"{video_id}/transcripts/audio_transcript_"))
        if not transcript_blobs:
            raise FileNotFoundError(f"No transcript file found for video {video_id}")
        
        transcript_blob = transcript_blobs[0]
        transcript_content = transcript_blob.download_as_text()
        transcript_data = json.loads(transcript_content)

        for result in transcript_data.get('results', []):
            for alternative in result.get('alternatives', []):
                transcript = alternative.get('transcript', '')
                plain_transcript += transcript + " "

                for word in alternative.get('words', []):
                    word_count += 1
                    confidence = word.get('confidence', 0)
                    total_confidence += confidence
                    json_transcript.append({
                        "start_time": word.get('startOffset', '0s').rstrip('s'),
                        "end_time": word.get('endOffset', '0s').rstrip('s'),
                        "word": word.get('word', ''),
                        "confidence": confidence
                    })

        # Clean up the plain transcript
        plain_transcript = plain_transcript.strip()

        # Calculate overall confidence score
        overall_confidence = total_confidence / word_count if word_count > 0 else 0

    except Exception as e:
        logger.error(f"Error processing transcription response for video {video_id}: {str(e)}", exc_info=True)
        plain_transcript = "Error processing transcription."
        json_transcript = []
        word_count = 0
        overall_confidence = 0

    return plain_transcript, json_transcript, word_count, overall_confidence

def process_batch(batch, video_id, bucket):
    try:
        uploads = []
        for frame, frame_number in batch:
            frame_filename = f'{frame_number:06d}.jpg'
            _, buffer = cv2.imencode('.jpg', frame)
            frame_blob = bucket.blob(f'{video_id}/frames/{frame_filename}')
            uploads.append((frame_blob, buffer.tobytes()))

        # Perform batch upload
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(uploads), 10)) as executor:
            list(executor.map(lambda x: x[0].upload_from_string(x[1], content_type='image/jpeg'), uploads))
        
    except Exception as e:
        logger.error(f"Error processing batch for video {video_id}: {str(e)}")

async def get_video_resolution(bucket: storage.Bucket, video_id: str) -> Tuple[int, int]:
    """
    Retrieve the video resolution by analyzing the first frame (000000.jpg) in the frames folder.
    """
    try:
        # Construct the path to the first frame
        first_frame_path = f'{video_id}/frames/000000.jpg'
        
        # Get the blob for the first frame
        frame_blob = bucket.blob(first_frame_path)
        
        # Check if the blob exists
        if not frame_blob.exists():
            raise FileNotFoundError(f"First frame not found for video {video_id}")
        
        # Download the frame data
        frame_data = frame_blob.download_as_bytes()
        
        # Open the image using PIL
        with Image.open(io.BytesIO(frame_data)) as img:
            width, height = img.size
        
        logger.info(f"Detected resolution for video {video_id}: {width}x{height}")
        return (width, height)
    
    except FileNotFoundError as e:
        logger.error(f"Error retrieving video resolution for {video_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving video resolution for {video_id}: {str(e)}")
        # Return a default resolution if unable to retrieve
        logger.warning(f"Using default resolution (1920x1080) for video {video_id}")
        return (1920, 1080)  # Default to 1080p

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)