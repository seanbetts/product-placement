import os
import uuid
import logging
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
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

async def process_video(video_id: str):
    logger.info(f"Starting to process video: {video_id}")
    try:
        # Your processing logic here
        logger.info(f"Completed processing video: {video_id}")
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)

@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)