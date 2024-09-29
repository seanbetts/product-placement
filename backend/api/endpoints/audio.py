from fastapi import APIRouter, HTTPException
from core.logging import logger
from services import audio_processing

router = APIRouter()

## TRANSCRIPT ENDPOINT (GET)
## Gets the transcript for a video
########################################################
@router.get("/{video_id}/transcript")
async def get_transcript(video_id: str):
    logger.debug(f"Received request for transcript of video: {video_id}")
    
    try:
        logger.debug(f"Attempting to retrieve transcript for video: {video_id}")
        transcript = await audio_processing.get_transcript(video_id)
        logger.debug(f"Successfully retrieved transcript for video {video_id}")
        return transcript

    except FileNotFoundError:
        logger.error(f"Transcript not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Transcript not found for video {video_id}")

    except Exception as e:
        logger.error(f"Error retrieving transcript for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving transcript")
########################################################