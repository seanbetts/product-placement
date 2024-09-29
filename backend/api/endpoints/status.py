from fastapi import APIRouter, HTTPException
from core.logging import logger
from services import status_processing
from utils.decorators import retry
from botocore.exceptions import ClientError 

router = APIRouter()

## STATUS ENDPOINT (GET)
## Returns status.json for a given video ID 
########################################################
@router.get("/{video_id}/video/status")
async def get_processing_status(video_id: str):
    logger.debug(f"Received request for processing status of video {video_id}")
    
    try:
        logger.debug(f"Attempting to retrieve processing status for video: {video_id}")
        
        @retry(exceptions=(ClientError,), tries=3, delay=1, backoff=2)
        async def get_status_with_retry():
            return await status_processing.get_processing_status(video_id)
        
        processing_status = await get_status_with_retry()
        
        logger.debug(f"Successfully retrieved processing status for video {video_id}")
        return processing_status
    
    except ClientError as e:
        logger.error(f"AWS client error after retries for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processing status of video from AWS")
    
    except FileNotFoundError:
        logger.warning(f"Status of video {video_id} not found")
        raise HTTPException(status_code=404, detail=f"Status of video {video_id} not found")
    
    except Exception as e:
        logger.error(f"Unexpected error retrieving processing status for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processing status of video")
######################################################## 

## PROCESSED VIDEOS ENDPOINT (GET)
## Returns a list of all processed videos
########################################################
@router.get("/video/processed-videos")
async def get_processed_videos():
    # Use the simpler AppLogger for this endpoint
    logger.debug("Received request for list of processed videos")
    
    try:
        logger.debug("Attempting to retrieve list of processed videos")
        processed_videos = await status_processing.get_processed_videos()
        logger.debug("Successfully retrieved list of processed videos")
        return processed_videos

    except FileNotFoundError:
        logger.error("List of processed videos not found")
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
    logger.debug(f"Received request for processing stats for video {video_id}")
    
    try:
        logger.debug(f"Attempting to retrieve processing stats for video {video_id}")
        processing_stats = await status_processing.get_processing_stats(video_id)
        logger.debug(f"Successfully retrieved processing stats for video {video_id}")
        return processing_stats

    except FileNotFoundError:
        logger.error(f"Processing stats not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Processing stats not found for video {video_id}")

    except Exception as e:
        logger.error(f"Error retrieving processing stats for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving processing stats for video {video_id}")
########################################################