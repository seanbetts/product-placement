import cv2
import os
import json
import tempfile
import uuid
import logging
import datetime
import time
import asyncio
import concurrent.futures
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import storage
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from dotenv import load_dotenv
from io import BytesIO
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up FastAPI
app = FastAPI()

PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BATCH_SIZE = 20
FRAME_INTERVAL = 1
STATUS_UPDATES = 100
MAX_WORKERS = 10

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
credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_APPLICATION_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
custom_transport = CustomTransport(credentials)
storage_client = storage.Client(credentials=credentials, _http=custom_transport)

@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "ok"}

@app.post("/upload")
async def upload_video(video: UploadFile, background_tasks: BackgroundTasks):
    logger.info(f"Received upload request for file: {video.filename}")
    video_id = str(uuid.uuid4())
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    blob = bucket.blob(f'{video_id}/original.mp4')
    
    try:
        logger.info(f"Uploading file to bucket: {PROCESSING_BUCKET}")
        blob.upload_from_file(video.file, content_type=video.content_type)
        logger.info(f"File uploaded successfully. Video ID: {video_id}")
        background_tasks.add_task(process_video, video_id)
        return {"video_id": video_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{video_id}")
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

async def process_video(video_id: str):
    logger.info(f"Starting to process video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    video_blob = bucket.blob(f'{video_id}/original.mp4')
    
    update_status(video_id, "processing", {})

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        video_blob.download_to_filename(temp_video.name)
        temp_video_path = temp_video.name

    try:
        start_time = time.time()
        start_datetime = datetime.datetime.now().isoformat()

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        frame_number = 0
        batches = []
        current_batch = []

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

            if frame_number % STATUS_UPDATES == 0:
                progress = (frame_number / frame_count) * 100
                update_status(video_id, "processing", {
                    "progress": f"{progress:.2f}%",
                    "frames_processed": f'{frame_number} frames of {frame_count} total frames'
                })
                logger.info(f"Processed {frame_number} of {frame_count} frames for video {video_id}")

        cap.release()

        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch, batch, video_id, bucket) for batch in batches]
            concurrent.futures.wait(futures)

        end_time = time.time()
        end_datetime = datetime.datetime.now().isoformat()
        processing_time = end_time - start_time
        frames_per_second = frame_number / processing_time
        processing_speed = (frames_per_second / fps) * 100

        stats = {
            "video_id": video_id,
            "video_length": f'{round(duration,1)} seconds',
            "total_frames": frame_count,
            "video_fps": round(fps, 2),
            "extracted_frames": frame_number // FRAME_INTERVAL,
            "processing_time": f'{processing_time:.2f} seconds',
            "processing_fps": f'{frames_per_second:.2f}',
            "processing_speed": f'{processing_speed:.1f}% of real-time',
            "processing_start_time": start_datetime,
            "processing_end_time": end_datetime
        }
        update_status(video_id, "complete", stats)
        
        logger.info(f"Completed processing video: {video_id}")
        logger.info(f"Extracted {frame_number // FRAME_INTERVAL} of {frame_count} frames from {video_id}")
        logger.info(f"Processing started at: {start_datetime}")
        logger.info(f"Processing ended at: {end_datetime}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Frames processed per second: {frames_per_second:.2f}")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        update_status(video_id, "error", {"error": str(e)})
    finally:
        os.unlink(temp_video_path)

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

def update_status(video_id: str, status: str, details: dict):
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    status_blob = bucket.blob(f'{video_id}/status.json')
    status_data = {
        "status": status,
        "last_updated": datetime.datetime.utcnow().isoformat(),
        "details": details
    }
    status_blob.upload_from_string(json.dumps(status_data), content_type='application/json')

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)