import uuid
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from core.logging import logger
from services import s3_operations, video_annotation, object_detection
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails

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

## PROCESS OBJECTS (POST)
## Processes objects data and combines with brand data
########################################################
@router.post("/process-objects/{video_id}")
async def process_objects_endpoint(video_id: str):
    logger.info(f"Received request to process objects data for video {video_id}")

    status_tracker = StatusTracker(video_id)
    await status_tracker.update_s3_status()

    video_details = await VideoDetails.create(video_id)

    brand_results = await s3_operations.get_brands_ocr_results(video_id)

    object_results = await s3_operations.get_objects_results(video_id)

    try:
        logger.debug(f"Starting processing of objects data for video: {video_id}")
        await object_detection.combine_object_and_brand_data(video_id, status_tracker, video_details, brand_results, object_results)
        logger.info(f"Object data processing completed for video {video_id}")
        return {"message": f"Object data processing completed for video {video_id}"}

    except Exception as e:
        logger.error(f"Error processing objects data for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running object detection: {str(e)}")
########################################################

## ANNOTATE VIDEOS (POST)
## Annotates videos with brand, logos, and object detection
########################################################
@router.post("/annotate_video/{video_id}")
async def annotate_video_endpoint(video_id: str):
    logger.info(f"Received request to annotate video {video_id}")
    status_tracker = StatusTracker(video_id)
    video_details = await VideoDetails.create(video_id)
    try:
        logger.info(f"Video Processing - Video Annotation - Step 5.1: Starting annotation and reconstruction for video: {video_id}")
        await video_annotation.annotate_video(video_id, status_tracker, video_details)
        logger.info(f"Video Processing - Video Annotation - Step 5.9: Video annotation and reconstruction completed for video {video_id}")
        return {"message": f"Video annotation completed for video_id: {video_id}"}
    except Exception as e:
        logger.error(f"Error annotating video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running video annotation: {str(e)}")
########################################################

## GET PROCESSED VIDEO ENDPOINT (GET)
## Returns the processed video file for a given video ID
########################################################
@router.get("/video/processed/{video_id}")
async def get_processed_video(video_id: str):
    logger.info(f"Received request to retrieve processed video for video_id: {video_id}")
    try:
        logger.debug(f"Attempting to generate pre-signed URL for video_id: {video_id}")
        video_url = await s3_operations.get_processed_video(video_id)
        
        if not video_url:
            logger.warning(f"Processed video not found for video_id: {video_id}")
            raise HTTPException(status_code=404, detail="Processed video not found")
        
        logger.debug(f"Successfully generated pre-signed URL for video_id: {video_id}")
        return JSONResponse(content={"url": video_url})
    
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error generating pre-signed URL for video_id {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processed video")
########################################################