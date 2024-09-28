from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from core.logging import video_logger, AppLogger, dual_log
from models.status_tracker import StatusTracker
from services import s3_operations, video_annotation, object_detection

# Create a global instance of AppLogger
app_logger = AppLogger()

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
    async with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _upload_video_endpoint():
            dual_log(vlogger, app_logger, 'info', "Received request to upload video for processing")
            
            try:
                vlogger.logger.info(f"Uploading chunk {chunk_number}/{total_chunks} for video ID: {video_id or 'new video'}")
                result = await s3_operations.upload_video(
                    vlogger,
                    background_tasks,
                    file,
                    chunk_number,
                    total_chunks,
                    video_id
                )
                vlogger.logger.info(f"Successfully uploaded video chunk {chunk_number}/{total_chunks} for video ID: {video_id or 'new video'}")
                return result

            except Exception as e:
                dual_log(vlogger, app_logger, 'error', f"Error processing video upload: {str(e)}", exc_info=True)
                return JSONResponse(status_code=500, content={"error": "Upload failed", "details": str(e)})

        return await _upload_video_endpoint()
########################################################

## CANCEL UPLOAD ENDPOINT (POST)
## Cancels an upload for a given video ID
########################################################
@router.post("/video/cancel-upload/{video_id}")
async def cancel_video_upload(video_id: str):
    async with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _cancel_video_upload():
            vlogger.logger.info(f"Received request to cancel upload for video {video_id}")

            try:
                vlogger.logger.info(f"Attempting to cancel upload for video: {video_id}")
                result = await vlogger.log_performance(s3_operations.cancel_upload_video)(vlogger, video_id)
                vlogger.logger.info(f"Successfully cancelled upload for video {video_id}")
                return {"status": "success", "message": f"Upload of video {video_id} cancelled"}

            except Exception as e:
                dual_log(vlogger, app_logger, 'error', f"Error cancelling upload for video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error cancelling video upload")

        return await _cancel_video_upload()
########################################################

## DETECT OBJECTS (POST)
## Detects objects in a video
########################################################
@router.post("/detect-objects/{video_id}")
async def detect_objects_endpoint(video_id: str):
    async with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _detect_objects_endpoint(v_id):
            vlogger.logger.info(f"Received request to detect objects in video {v_id}")
            status_tracker = StatusTracker(v_id)

            try:
                vlogger.logger.info(f"Starting detecting objects in video: {v_id}")
                await vlogger.log_performance(object_detection.detect_objects)(vlogger, v_id, status_tracker)
                vlogger.logger.info(f"Object detection completed in video {v_id}")
                return {"message": f"Object detection completed in video {v_id}"}

            except Exception as e:
                dual_log(vlogger, app_logger, 'info', f"Error detecting objects in video {v_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error running object detection: {str(e)}")

        return await _detect_objects_endpoint(video_id)
########################################################

## ANNOTATE VIDEOS (POST)
## Annotates videos with brand, logos, and object detection
########################################################

@router.post("/annotate_video/{video_id}")
async def annotate_video_endpoint(video_id: str):
    async with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _annotate_video_endpoint(v_id):
            dual_log(vlogger, app_logger, 'info', f"Received request to annotate video {v_id}")
            status_tracker = StatusTracker(v_id)
            try:
                vlogger.logger.info(f"Starting annotation for video: {v_id}")
                await video_annotation.annotate_video(vlogger, v_id, status_tracker)
                vlogger.logger.info(f"Video annotation completed for video {v_id}")
                return {"message": f"Video annotation completed for video_id: {v_id}"}
            except Exception as e:
                dual_log(vlogger, app_logger, 'info', f"Error annotating video {v_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error running video annotation: {str(e)}")

        return await _annotate_video_endpoint(video_id)
########################################################