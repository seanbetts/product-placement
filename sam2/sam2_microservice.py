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
        config_dir = "sam2_configs"
        config_name = "sam2_hiera_l.yaml"
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Config directory: {config_dir}")
        logger.info(f"Config name: {config_name}")
        logger.info(f"Config file exists: {os.path.exists(os.path.join(config_dir, config_name))}")
        logger.info(f"Contents of {config_dir}: {os.listdir(config_dir)}")
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

## SEGMENT VIDEO ENDPOINT (POST)
## Segments a video and returns the masks
@app.post("/segment_video")
async def segment_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name

        background_tasks.add_task(process_video, temp_video_path)

        return JSONResponse(content={"message": "Video processing started"})
    except Exception as e:
        logger.error(f"Error in segment_video: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
########################################################

########################################################
## FUNCTIONS                                          ##
########################################################

## PROCESS VIDEO FUNCTION
## Processes a video and returns the masks
async def process_video(video_path: str):
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = video_predictor.init_state(video_path)
            
            results = []
            for frame_idx, object_ids, mask_logits in video_predictor.propagate_in_video(state):
                results.append({
                    "frame_idx": frame_idx,
                    "object_ids": object_ids.tolist(),
                    "masks": mask_logits.cpu().numpy().tolist()
                })

        with open(f"{os.path.splitext(video_path)[0]}_sam2_result.json", "w") as f:
            json.dump(results, f)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
    finally:
        os.unlink(video_path)
########################################################

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)