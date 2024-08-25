import cv2
import os
import json
import tempfile
import uuid
import logging
import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up FastAPI
logger.info("Initializing FastAPI application...")
app = FastAPI()
logger.info("FastAPI application initialized.")

storage_client = storage.Client()
PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')

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
    
    # Update status to "processing"
    update_status(video_id, "processing", {})

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        # Download video to a temporary file
        video_blob.download_to_filename(temp_video.name)
        temp_video_path = temp_video.name

    try:
        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f'{frame_number:06d}.jpg'
            _, buffer = cv2.imencode('.jpg', frame)
            frame_blob = bucket.blob(f'{video_id}/frames/{frame_filename}')
            frame_blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

            frame_number += 1

            # Update status every 100 frames
            if frame_number % 100 == 0:
                progress = (frame_number / frame_count) * 100
                update_status(video_id, "processing", {
                    "progress": f"{progress:.2f}%",
                    "frames_processed": frame_number,
                    "total_frames": frame_count
                })
                logger.info(f"Processed {frame_number} frames for video {video_id}")

        cap.release()

        # Save final statistics
        stats = {
            "video_id": video_id,
            "video_length": f'{round(duration,1)} seconds',
            "total_frames": frame_count,
            "extracted_frames": frame_number,
            "fps": round(fps, 2)
        }
        update_status(video_id, "complete", stats)
        
        logger.info(f"Completed processing video: {video_id}")
        logger.info(f"Extracted {frame_number} frames from {video_id}")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        update_status(video_id, "error", {"error": str(e)})
    finally:
        # Clean up the temporary file
        os.unlink(temp_video_path)

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