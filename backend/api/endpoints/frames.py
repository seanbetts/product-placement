from fastapi import APIRouter, HTTPException
from core.logging import logger
from services import frames_processing

router = APIRouter()

## FIRST VIDEO FRAME ENDPOINT (GET)   
## Returns the first frame of the video as a JPEG image
########################################################
@router.get("/{video_id}/images/first-frame")
async def get_first_video_frame(video_id: str):
    # logger.info(f"Received request for first frame of video: {video_id}")
    try:
        first_frame = await frames_processing.get_first_video_frame(video_id)
        return first_frame
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"First frame not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving first frame for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving first frame")
########################################################

## ALL VIDEO FRAMES ENDPOINT (GET)
## Returns a list of all video frames
########################################################
@router.get("/{video_id}/images/all-frames")
async def get_all_video_frames(video_id: str):
    # logger.info(f"Received request for all frames of video: {video_id}")
    try:
        all_frames = await frames_processing.get_all_video_frames(video_id)
        return all_frames
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"All frames not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving all frames for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving all frames")
########################################################