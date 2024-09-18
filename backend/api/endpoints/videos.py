from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from core.logging import logger
from services import s3_operations
from services import video_post_processing

router = APIRouter()

## UPLOAD ENDPOINT (POST)
## Uploads a video to the processing bucket and schedules the video processing
########################################################
@router.post("/video/upload")
async def upload_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    video_id: Optional[str] = Form(None)
):
    logger.info(f"Received request to upload video for processing")
    try:
        result = await s3_operations.upload_video(
            background_tasks,
            file,
            chunk_number,
            total_chunks,
            video_id
        )
        return result
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Upload failed", "details": str(e)})
########################################################

## CANCEL UPLOAD ENDPOINT (POST)
## Cancels an upload for a given video ID
########################################################
@router.post("/video/cancel-upload/{video_id}")
async def cancel_video_upload(video_id):
    logger.info(f"Received request to cancel upload for video {video_id}")
    try:
        result = await s3_operations.cancel_upload_video(video_id)
        return {"status": "success", "message": f"Upload of video {video_id} cancelled"}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail="Error cancelling video upload")
########################################################

## POST-PROCESS VIDEOS (POST)
## Annotates videos with brand, logos, and object detection
########################################################
@router.post("/process_video/{video_id}")
async def process_video_endpoint(video_id: str):
    logger.info(f"Received request to run post-processing on video {video_id}")
    try:
        video_post_processing.process_video_frames(video_id)
        return {"message": f"Video post-processing initiated for video_id: {video_id}"}
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running video post-processing: {str(e)}")
########################################################