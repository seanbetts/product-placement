from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from core.logging import AppLogger
from services import s3_operations

# Create a global instance of AppLogger
app_logger = AppLogger()

router = APIRouter()

## DOWNLOAD ENDPOINT (GET)
## Downloads a file for a given video ID and file type
########################################################
@router.get("/{video_id}/files/download/{file_type}")
async def download_file(video_id: str, file_type: str):
    # app_logger.log_info(f"Received request for download of {file_type} for video: {video_id}")
    
    try:
        # app_logger.log_info(f"Attempting to download {file_type} for video: {video_id}")
        url = await s3_operations.download_file_from_s3(video_id, file_type)
        # app_logger.log_info(f"Successfully retrieved {file_type} for video {video_id}")
        return RedirectResponse(url=url)

    except FileNotFoundError:
        app_logger.log_error(f"File not found: {file_type} for video {video_id}")
        raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        app_logger.log_error(f"Error retrieving {file_type} for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving {file_type} for video")
########################################################