import uuid
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from core.logging import logger
from services import s3_operations, video_annotation, object_detection

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
    if not video_id:
        video_id = str(uuid.uuid4())

    logger.info("Received request to upload video for processing")
    
    try:
        logger.debug(f"Uploading chunk {chunk_number}/{total_chunks} for video ID: {video_id or 'new video'}")
        result = await s3_operations.upload_video(
            background_tasks,
            file,
            chunk_number,
            total_chunks,
            video_id
        )
        logger.debug(f"Successfully uploaded video chunk {chunk_number}/{total_chunks} for video ID: {video_id or 'new video'}")
        return result

    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Upload failed", "details": str(e)})
########################################################

## CANCEL UPLOAD ENDPOINT (POST)
## Cancels an upload for a given video ID
########################################################
@router.post("/video/cancel-upload/{video_id}")
async def cancel_video_upload(video_id: str):
    logger.info(f"Received request to cancel upload for video {video_id}")
    try:
        logger.debug(f"Attempting to cancel upload for video: {video_id}")
        result = await s3_operations.cancel_upload_video(video_id)
        logger.info(f"Successfully cancelled upload for video {video_id}")
        return {"status": "success", "message": f"Upload of video {video_id} cancelled"}

    except Exception as e:
        logger.error(f"Error cancelling upload for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error cancelling video upload")
########################################################

## DETECT OBJECTS (POST)
## Detects objects in a video
########################################################
@router.post("/detect-objects/{video_id}")
async def detect_objects_endpoint(video_id: str):
    logger.info(f"Received request to detect objects in video {video_id}")
    try:
        logger.debug(f"Starting detecting objects in video: {video_id}")
        await object_detection.detect_objects(video_id)
        logger.info(f"Object detection completed in video {video_id}")
        return {"message": f"Object detection completed in video {video_id}"}

    except Exception as e:
        logger.error(f"Error detecting objects in video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running object detection: {str(e)}")
########################################################

## ANNOTATE VIDEOS (POST)
## Annotates videos with brand, logos, and object detection
########################################################
@router.post("/annotate_video/{video_id}")
async def annotate_video_endpoint(video_id: str):
    logger.info(f"Received request to annotate video {video_id}")
    try:
        logger.debug(f"Starting annotation for video: {video_id}")
        await video_annotation.annotate_video(video_id)
        logger.info(f"Video annotation completed for video {video_id}")
        return {"message": f"Video annotation completed for video_id: {video_id}"}
    except Exception as e:
        logger.error(f"Error annotating video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running video annotation: {str(e)}")
########################################################