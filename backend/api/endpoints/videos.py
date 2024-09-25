from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from core.logging import video_logger, AppLogger, dual_log
from models.status_tracker import StatusTracker
from services import s3_operations
from services import video_post_processing

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
    with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _upload_video_endpoint():
            dual_log(vlogger, app_logger, 'info', "Received request to upload video for processing")
            
            try:
                vlogger.logger.debug(f"Uploading chunk {chunk_number}/{total_chunks} for video ID: {video_id or 'new video'}")
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
    with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _cancel_video_upload():
            vlogger.logger.info(f"Received request to cancel upload for video {video_id}")

            try:
                vlogger.logger.debug(f"Attempting to cancel upload for video: {video_id}")
                result = await vlogger.log_performance(s3_operations.cancel_upload_video)(vlogger, video_id)
                vlogger.logger.info(f"Successfully cancelled upload for video {video_id}")
                return {"status": "success", "message": f"Upload of video {video_id} cancelled"}

            except Exception as e:
                dual_log(vlogger, app_logger, 'error', f"Error cancelling upload for video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error cancelling video upload")

        return await _cancel_video_upload()
########################################################

## ANNOTATE VIDEOS (POST)
## Annotates videos with brand, logos, and object detection
########################################################
@router.post("/annotate_video/{video_id}")
async def annotate_video_endpoint(video_id: str):
    with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _annotate_video_endpoint():
            vlogger.logger.info(f"Received request to annotate video {video_id}")
            status_tracker = StatusTracker(video_id)

            try:
                vlogger.logger.debug(f"Starting annotation for video: {video_id}")
                await vlogger.log_performance(video_post_processing.annotate_video)(vlogger, video_id, status_tracker)
                vlogger.logger.info(f"Video annotationg completed for video {video_id}")
                return {"message": f"Video annotation completed for video_id: {video_id}"}

            except Exception as e:
                vlogger.logger.error(f"Error annotating video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error running video annotation: {str(e)}")

        return await _annotate_video_endpoint()
########################################################