from fastapi import APIRouter, HTTPException
from core.logging import AppLogger
from services import audio_processing

# Create a global instance of AppLogger
app_logger = AppLogger()

router = APIRouter()

## TRANSCRIPT ENDPOINT (GET)
## Gets the transcript for a video
########################################################
@router.get("/{video_id}/transcript")
async def get_transcript(video_id: str):
    # app_logger.log_info(f"Received request for transcript of video: {video_id}")
    
    try:
        # app_logger.log_info(f"Attempting to retrieve transcript for video: {video_id}")
        transcript = await audio_processing.get_transcript(video_id)
        # app_logger.log_info(f"Successfully retrieved transcript for video {video_id}")
        return transcript

    except FileNotFoundError:
        app_logger.log_error(f"Transcript not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Transcript not found for video {video_id}")

    except Exception as e:
        app_logger.log_error(f"Error retrieving transcript for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving transcript")
########################################################