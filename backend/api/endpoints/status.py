from fastapi import APIRouter, HTTPException
from core.logging import logger
from services import status_processing

router = APIRouter()

## STATUS ENDPOINT (GET)
## Returns status.json for a given video ID 
########################################################
@router.get("/{video_id}/video/status")
async def get_processing_status(video_id):
    logger.info(f"Received request for list of processed videos")
    try:
        processing_status = await status_processing.get_processing_status(video_id)
        return processing_status
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Status of video {video_id} not found")
    except Exception as e:
        logger.error(f"Error retrieving list of processed videos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving list of processed videos")
######################################################## 

## PROCESSED VIDEOS ENDPOINT (GET)
## Returns a list of all processed videos
########################################################
@router.get("/video/processed-videos")
async def get_processed_videos():
    logger.info(f"Received request for list of processed videos")
    try:
        processed_videos = await status_processing.get_processed_videos()
        return processed_videos
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="List of processed videos not found")
    except Exception as e:
        logger.error(f"Error retrieving list of processed videos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving list of processed videos")
########################################################

## PROCESSING STATS ENDPOINT (GET)
## Returns the processing_stats.json for a given video ID
########################################################
@router.get("/{video_id}/video/processing-stats")
async def get_processing_stats(video_id: str):
    logger.info(f"Received request for processing stats for video: {video_id}")
    try:
        transcript = await status_processing.get_processing_stats(video_id)
        return transcript
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Processing stats not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving processing stats for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processing stats")
########################################################