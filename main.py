import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from google.cloud import storage
import uuid

# Load environment variables from .env file
load_dotenv()

# Access environment variables
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')

app = FastAPI()
storage_client = storage.Client()

@app.post("/upload")
async def upload_video(video: UploadFile = File(...), background_tasks: BackgroundTasks):
    video_id = str(uuid.uuid4())
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    blob = bucket.blob(f'{video_id}/original.mp4')
    
    blob.upload_from_file(video.file, content_type=video.content_type)
    
    background_tasks.add_task(process_video, video_id)
    
    return {"video_id": video_id, "status": "processing"}

async def process_video(video_id: str):
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    
    # Download the original video
    blob = bucket.blob(f'{video_id}/original.mp4')
    local_path = f'/tmp/{video_id}_original.mp4'
    blob.download_to_filename(local_path)
    
    # Process video (placeholder for SAM 2 logic)
    # This is where you'd implement frame extraction, object detection, and tracking
    
    # Upload processed frames (example)
    for i in range(0, 1000, 50):  # Assuming 1000 frames, processing every 50th
        frame_blob = bucket.blob(f'{video_id}/frames/frame_{i:04d}.jpg')
        # In reality, you'd be saving actual processed frames here
        frame_blob.upload_from_string(f"Processed frame {i}", content_type='image/jpeg')
    
    # Upload statistics (placeholder)
    stats_blob = bucket.blob(f'{video_id}/statistics.json')
    stats_blob.upload_from_string('{"placeholder": "statistics"}', content_type='application/json')
    
    print(f"Completed processing video {video_id}")

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    stats_blob = bucket.blob(f'{video_id}/statistics.json')
    
    if stats_blob.exists():
        return {"video_id": video_id, "status": "complete"}
    else:
        return {"video_id": video_id, "status": "processing"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)