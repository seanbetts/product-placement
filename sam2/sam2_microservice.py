import os
import io
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from google.cloud import storage
from PIL import Image
import torch
import tempfile
from segment_anything_2 import build_sam2, SAM2ImagePredictor

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

#Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI server...")

app = FastAPI()

# Initialize SAM 2 model
checkpoint = "/app/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Initialize Google Cloud Storage client
storage_client = storage.Client()

########################################################
## IMAGE ENDPOINTS                                    ##
########################################################

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
@app.get("/health")
async def health_check():
    logger.debug("Health check called")
    return {"status": "ok"}
########################################################

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL Image to numpy array
        image_array = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Process the image with SAM 2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_array)
            masks, _, _ = predictor.predict()
        
        # Convert masks to a format suitable for JSON
        masks_list = masks.tolist()
        
        return JSONResponse(content={"masks": masks_list})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/segment_gcs")
async def segment_image_gcs(bucket_name: str, blob_name: str):
    try:
        # Download the image from Google Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to numpy array
        image_array = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Process the image with SAM 2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_array)
            masks, _, _ = predictor.predict()
        
        # Convert masks to a format suitable for JSON
        masks_list = masks.tolist()
        
        # Save results back to GCS
        result_blob_name = f"{os.path.splitext(blob_name)[0]}_sam2_result.json"
        result_blob = bucket.blob(result_blob_name)
        result_blob.upload_from_string(
            json.dumps({"masks": masks_list}),
            content_type="application/json"
        )
        
        return JSONResponse(content={"result_blob": result_blob_name})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

########################################################
## VIDEO ENDPOINTS                                    ##
########################################################

@app.post("/segment_video")
async def segment_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name

        # Start background task for video processing
        background_tasks.add_task(process_video, temp_video_path)

        return JSONResponse(content={"message": "Video processing started"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/segment_video_gcs")
async def segment_video_gcs(bucket_name: str, blob_name: str, background_tasks: BackgroundTasks):
    try:
        # Download the video from Google Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            blob.download_to_filename(temp_video.name)
            temp_video_path = temp_video.name

        # Start background task for video processing
        background_tasks.add_task(process_video_gcs, temp_video_path, bucket_name, blob_name)

        return JSONResponse(content={"message": "Video processing started"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


########################################################
## FUNCTIONS                                          ##
########################################################

async def process_video(video_path: str):
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path)
            
            results = []
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                results.append({
                    "frame_idx": frame_idx,
                    "object_ids": object_ids.tolist(),
                    "masks": masks.tolist()
                })

        # Save results (you might want to adjust this based on your needs)
        with open(f"{os.path.splitext(video_path)[0]}_sam2_result.json", "w") as f:
            json.dump(results, f)

    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(video_path)

async def process_video_gcs(video_path: str, bucket_name: str, blob_name: str):
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path)
            
            results = []
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                results.append({
                    "frame_idx": frame_idx,
                    "object_ids": object_ids.tolist(),
                    "masks": masks.tolist()
                })

        # Save results back to GCS
        bucket = storage_client.bucket(bucket_name)
        result_blob_name = f"{os.path.splitext(blob_name)[0]}_sam2_result.json"
        result_blob = bucket.blob(result_blob_name)
        result_blob.upload_from_string(
            json.dumps(results),
            content_type="application/json"
        )

    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(video_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)