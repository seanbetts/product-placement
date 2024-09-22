from fastapi import APIRouter, HTTPException
from core.logging import video_logger, AppLogger
from services import status_processing

# Create a global instance of AppLogger
app_logger = AppLogger()

router = APIRouter()

## STATUS ENDPOINT (GET)
## Returns status.json for a given video ID 
########################################################
@router.get("/{video_id}/video/status")
async def get_processing_status(video_id: str):
    with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _get_processing_status():
            vlogger.logger.info(f"Received request for processing status of video {video_id}")

            try:
                vlogger.logger.debug(f"Attempting to retrieve processing status for video: {video_id}")
                processing_status = await vlogger.log_performance(status_processing.get_processing_status)(vlogger, video_id)
                vlogger.logger.info(f"Successfully retrieved processing status for video {video_id}")
                return processing_status

            except FileNotFoundError:
                vlogger.logger.warning(f"Status of video {video_id} not found")
                raise HTTPException(status_code=404, detail=f"Status of video {video_id} not found")

            except Exception as e:
                vlogger.logger.error(f"Error retrieving processing status for video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error retrieving processing status of video")

        return await _get_processing_status()
######################################################## 

## PROCESSED VIDEOS ENDPOINT (GET)
## Returns a list of all processed videos
########################################################
@router.get("/video/processed-videos")
async def get_processed_videos():
    # Use the simpler AppLogger for this endpoint
    # app_logger.log_info("Received request for list of processed videos")
    
    try:
        # app_logger.log_info("Attempting to retrieve list of processed videos")
        processed_videos = await status_processing.get_processed_videos()
        # app_logger.log_info("Successfully retrieved list of processed videos")
        return processed_videos

    except FileNotFoundError:
        app_logger.log_error("List of processed videos not found")
        raise HTTPException(status_code=404, detail="List of processed videos not found")

    except Exception as e:
        app_logger.log_error(f"Error retrieving list of processed videos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving list of processed videos")
########################################################

## PROCESSING STATS ENDPOINT (GET)
## Returns the processing_stats.json for a given video ID
########################################################
@router.get("/{video_id}/video/processing-stats")
async def get_processing_stats(video_id: str):
    # app_logger.log_info(f"Received request for processing stats for video {video_id}")
    
    try:
        # app_logger.log_info(f"Attempting to retrieve processing stats for video {video_id}")
        processing_stats = await status_processing.get_processing_stats(video_id)
        # app_logger.log_info(f"Successfully retrieved processing stats for video {video_id}")
        return processing_stats

    except FileNotFoundError:
        app_logger.log_error(f"Processing stats not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Processing stats not found for video {video_id}")

    except Exception as e:
        app_logger.log_error(f"Error retrieving processing stats for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving processing stats for video {video_id}")
########################################################