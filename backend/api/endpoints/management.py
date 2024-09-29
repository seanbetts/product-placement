from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.api import NameUpdate
from core.logging import logger
from services import video_processing
from utils import utils

router = APIRouter()

## UPDATE VIDEO NAME ENDPOINT (POST)
## Updates the name of a video
########################################################
@router.post("/{video_id}/video/update-name")
async def update_video_name(video_id: str, name_update: NameUpdate):
    logger.debug(f"Received request to update name for video {video_id} to '{name_update.name}'")
    
    try:
        logger.debug(f"Attempting to update name for video: {video_id}")
        result = await utils.update_video_name(video_id, name_update.name)
        logger.debug(f"Successfully updated name for video {video_id} to '{name_update.name}'")
        return {"message": f"Video name updated successfully to {name_update.name}"}

    except FileNotFoundError:
        logger.error(f"Video {video_id} not found")
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    except Exception as e:
        logger.error(f"Error updating name for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating video name")
########################################################

## DELETE VIDEO ENDPOINT (DELETE)
## Deletes a video
########################################################
@router.delete("/{video_id}/video")
async def delete_video(video_id: str, background_tasks: BackgroundTasks):
    logger.debug(f"Received request to delete video: {video_id}")

    try:
        # Start the deletion process
        logger.debug(f"Starting background task for video deletion: {video_id}")
        background_tasks.add_task(video_processing.delete_video, video_id)
        logger.debug(f"Video deletion process started for video ID: {video_id}")
        return {"message": f"Video deletion process started for video ID: {video_id}"}

    except Exception as e:
        logger.error(f"Error initiating deletion for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error initiating video deletion")
########################################################
