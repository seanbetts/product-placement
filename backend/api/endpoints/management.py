from fastapi import APIRouter, HTTPException, BackgroundTasks
from core.logging import logger
from models.api import NameUpdate
from services import video_processing
from utils import utils

router = APIRouter()

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
########################################################
@router.get("/health")
async def health_check():
    logger.debug("Health check called")
    return {"status": "product placement backend ok"}
########################################################

## UPDATE VIDEO NAME ENDPOINT (POST)
## Updates the name of a video
########################################################
@router.post("/{video_id}/video/update-name")
async def update_video_name(video_id: str, name_update: NameUpdate):
    logger.info(f"Received request to update name for video {video_id} to '{name_update.name}'")
    try:
        result = await utils.update_video_name(video_id, name_update.name)
        return {"message": f"Video name updated successfully to {name_update.name}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    except Exception as e:
        logger.error(f"Error updating name for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating video name")
########################################################

## DELETE VIDEO ENDPOINT (DELETE)
## Deletes a video
########################################################
@router.delete("/{video_id}/video")
async def delete_video(video_id: str, background_tasks: BackgroundTasks):
    logger.info(f"Received request to delete video: {video_id}")
    try:
        # Start the deletion process
        background_tasks.add_task(video_processing.delete_video, video_id)
        return {"message": f"Video deletion process started for video ID: {video_id}"}
    except Exception as e:
        logger.error(f"Error initiating deletion for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initiating video deletion")
########################################################
