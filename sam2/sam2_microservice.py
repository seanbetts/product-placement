import os
import sys
import io
import json
import logging
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from google.cloud import storage
from google.oauth2 import service_account
from PIL import Image
import torch
import shutil
import tempfile
import numpy as np
import cv2

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI server...")

app = FastAPI()

# Initialize Google Cloud Storage client
try:
    credentials = service_account.Credentials.from_service_account_file('keyfile.json')
    storage_client = storage.Client(credentials=credentials)
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud Storage client: {str(e)}")
    raise

# Global variables to hold the SAM2 models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = "/app/checkpoints/sam2_hiera_large.pt"
config = "sam2_hiera_l.yaml"

# Initialize SAM2 models
sam2_model = None
image_predictor = None
mask_generator = None
video_predictor = None

########################################################
## STARTUP PROCESSES                                  ##
########################################################

## Initialize SAM2 models
@app.on_event("startup")
async def startup_event():
    global sam2_model, image_predictor, mask_generator, video_predictor
    try:
        logger.info(f"Device: {device}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            
        logger.info("Initializing SAM2 model...")
        sam2_model = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)
    
        # Initialize SAM2 image predictor
        logger.info("Initializing SAM2 image predictor...")
        image_predictor = SAM2ImagePredictor(sam2_model)
        
        # Initialize SAM2 automatic mask generator
        logger.info("Initializing SAM2 automatic mask generator...")
        mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

        # Initialize SAM2 video predictor
        logger.info("Initializing SAM2 video predictor...")
        video_predictor = build_sam2_video_predictor(config, checkpoint)

        logger.info("SAM2 models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SAM2 models: {str(e)}")
        logger.exception("Traceback:")
        raise

########################################################

########################################################
## FAST API IMAGE ENDPOINTS                           ##
########################################################

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
@app.get("/health")
async def health_check():
    logger.debug("Health check called")
    return {"status": "sam2 backend ok"}
########################################################

## SEGMENT IMAGE ENDPOINT (POST)
## Segments an image and returns the masks  
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            image_predictor.set_image(image_rgb)
            masks, _, _ = image_predictor.predict()
        
        masks_list = masks.tolist()
        
        return JSONResponse(content={"masks": masks_list})
    except Exception as e:
        logger.error(f"Error in segment_image: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
########################################################

########################################################
## FAST API VIDEO ENDPOINTS                           ##
########################################################

## SEGMENT VIDEO GCS ENDPOINT (POST)
## Segments a video from GCS and returns the masks
@app.post("/segment_video_gcs/{video_id}")
async def segment_video_gcs(background_tasks: BackgroundTasks, video_id: str):
    try:
        bucket_name = os.getenv('PROCESSING_BUCKET')
        frames_prefix = f"{video_id}/frames/"
        
        # Check if frames exist
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=frames_prefix))
        
        if not blobs:
            return JSONResponse(status_code=404, content={"error": f"Frames for video with id {video_id} not found"})

        # Process the frames
        background_tasks.add_task(process_video_frames, bucket_name, video_id)

        return JSONResponse(content={"message": f"Video processing started for video_id: {video_id}"})
    except Exception as e:
        logger.error(f"Error in segment_video_gcs: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
########################################################

########################################################
## FUNCTIONS                                          ##
########################################################

## PROCESS VIDEO FUNCTION
## Processes a video and returns the masks
async def process_video_frames(bucket_name: str, video_id: str):
    temp_dir = None
    try:
        # Create a temporary directory to store frames
        temp_dir = tempfile.mkdtemp()
        frames_prefix = f"{video_id}/frames/"
        
        # Download frames from GCS
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=frames_prefix)
        for blob in blobs:
            if blob.name.lower().endswith('.jpg'):
                local_path = os.path.join(temp_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
        
        logger.info(f"Downloaded frames for video {video_id} to {temp_dir}")
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = video_predictor.init_state(temp_dir)
            
            results = []
            for frame_idx, object_ids, mask_logits in video_predictor.propagate_in_video(state):
                results.append({
                    "frame_idx": frame_idx,
                    "object_ids": object_ids.tolist(),
                    "masks": mask_logits.cpu().numpy().tolist()
                })

        result_json = json.dumps(results)

        # Upload to GCS
        result_blob_name = f"{video_id}/sam2_result.json"
        blob = bucket.blob(result_blob_name)
        blob.upload_from_string(result_json, content_type='application/json')

        logger.info(f"Completed processing for video {video_id}")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
########################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)