import json
import traceback
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from core.logging import video_logger, AppLogger, dual_log
from core.s3_download import get_s3_object
from services import s3_operations
from services.ocr_processing import main_ocr_processing
from utils.utils import get_video_resolution

# Create a global instance of AppLogger
app_logger = AppLogger()

router = APIRouter()

## REPROCESS OCR ENDPOINT (POST)
## Reprocesses the raw OCR data for the video
########################################################
@router.post("/{video_id}/ocr/reprocess-ocr")
async def reprocess_ocr(video_id: str) -> Dict[str, Any]:
    async with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _reprocess_ocr(vlogger, v_id: str) -> Dict[str, Any]:
            dual_log(vlogger, app_logger, 'info', f"Received request to reprocess OCR for video {v_id}")
            video_details = await VideoDetails.create(video_id)
            status_tracker = StatusTracker(v_id)
            await status_tracker.update_s3_status()
            
            try:
                # Fetch processing stats from S3
                # vlogger.logger.debug(f"About to fetch processing stats for video {v_id}")
                try:
                    stats_obj = await get_s3_object(vlogger, f'{v_id}/processing_stats.json')
                    # vlogger.logger.debug(f"Successfully retrieved processing stats for video {v_id}")
                except Exception as e:
                    # vlogger.logger.debug(f"Error fetching processing stats for video {v_id}: {str(e)}")
                    raise HTTPException(status_code=500, detail="Error fetching processing stats")

                if stats_obj is None:
                    # vlogger.logger.debug(f"Processing stats not found for video {v_id}")
                    raise HTTPException(status_code=404, detail="Processing stats not found")

                stats = json.loads(stats_obj.decode('utf-8'))
                
                # Set video details
                try:
                    video_resolution = await get_video_resolution(vlogger, v_id)
                    fps = stats['video']['video_fps']
                    frame_count = stats['video']['total_frames']
                    duration = frame_count / fps

                    video_details.set_detail("video_resolution", video_resolution)
                    video_details.set_detail("frames_per_second", fps)
                    video_details.set_detail("number_of_frames", frame_count)
                    video_details.set_detail("video_length", duration)
                    
                    # vlogger.logger.debug(f"Video details - Resolution: {video_resolution}, FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f} seconds")
                except KeyError as e:
                    dual_log(vlogger, app_logger, 'error', f"Missing key in stats for video {v_id}: {str(e)}")
                    raise HTTPException(status_code=500, detail="Error retrieving video details")
                
                # Start OCR reprocessing
                dual_log(vlogger, app_logger, 'info', f"Starting to reprocess OCR data for video: {v_id} with resolution: {video_resolution}")
                result = await vlogger.log_performance(main_ocr_processing.post_process_ocr)(vlogger, v_id, status_tracker, video_details)
                dual_log(vlogger, app_logger, 'info', f"Reprocessing of OCR Data for video {v_id} complete, identified {len(result)} brands")
                
                return {"status": "success", "message": f"OCR data reprocessing completed, identified {len(result)} brands"}
            except Exception as e:
                dual_log(vlogger, app_logger, 'error', f"Error reprocessing OCR data for video {v_id}: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail="Error reprocessing OCR data")

        return await _reprocess_ocr(vlogger, video_id)
########################################################

## WORDCLOUD ENDPOINT (GET)
## Returns the wordcloud for a video
########################################################
@router.get("/{video_id}/ocr/wordcloud")
async def get_word_cloud(video_id: str):
    # app_logger.log_info(f"Received request for word cloud of video: {video_id}")
    try:
        # app_logger.log_info(f"Attempting to retrieve word cloud for video: {video_id}")
        wordcloud = await s3_operations.get_word_cloud(video_id)
        # app_logger.log_info(f"Successfully retrieved word cloud for video {video_id}")
        return wordcloud

    except FileNotFoundError:
        app_logger.error(f"Word cloud not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Word cloud not found")

    except Exception as e:
        app_logger.log_error(f"Error retrieving word cloud for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## BRANDS OCR TABLE ENDPOINT (GET)
## Returns the brands OCR table for a video
########################################################
@router.get("/{video_id}/ocr/brands-ocr-table")
async def get_brands_ocr_table(video_id: str):
    # app_logger.log_info(f"Received request for brand OCR table of video: {video_id}")
    try:
        # app_logger.log_info(f"Attempting to retrieve brand OCR table for video: {video_id}")
        table = await s3_operations.get_brands_ocr_table(video_id)
        # app_logger.log_info(f"Successfully retrieved brand OCR table for video {video_id}")
        return table

    except FileNotFoundError:
        app_logger.log_error(f"Brands OCR table not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Brands OCR table not found")

    except Exception as e:
        app_logger.log_error(f"Error retrieving brands OCR table for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brands OCR table")
########################################################

## OCR RESULTS ENDPOINT (GET)
## Returns the OCR results for a video
########################################################
@router.get("/{video_id}/ocr/results")
async def get_ocr_results(video_id: str):
    # app_logger.log_info(f"Received request for OCR results of video: {video_id}")
    try:
        # app_logger.log_info(f"Attempting to retrieve OCR results for video: {video_id}")
        results = await s3_operations.get_ocr_results(video_id)
        # app_logger.log_info(f"Successfully retrieved OCR results for video {video_id}")
        return results

    except FileNotFoundError:
        app_logger.log_error(f"OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail="OCR results not found")

    except Exception as e:
        app_logger.log_error(f"Error retrieving OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving OCR results")
########################################################

## PROCESSED OCR DATA ENDPOINT (GET)
## Returns the processed OCR results for a video
########################################################
@router.get("/{video_id}/ocr/processed-ocr")
async def get_processed_ocr(video_id: str):
    # app_logger.log_info(f"Received request for processed OCR results for video: {video_id}")
    try:
        # app_logger.log_info(f"Attempting to retrieve processed OCR results for video: {video_id}")
        processed_ocr = await s3_operations.get_processed_ocr_results(video_id)
        # app_logger.log_info(f"Successfully retrieved processed OCR results for video {video_id}")
        return processed_ocr

    except FileNotFoundError:
        app_logger.log_error(f"Processed OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Processed OCR results not found for video {video_id}")

    except Exception as e:
        app_logger.log_error(f"Error retrieving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results")
########################################################

## BRANDS OCR DATA ENDPOINT (GET)
## Returns the brands OCR results for a video
########################################################
@router.get("/{video_id}/ocr/brands-ocr")
async def get_brands_ocr(video_id: str):
    # app_logger.log_info(f"Received request for brands OCR results for video: {video_id}")
    try:
        # app_logger.log_info(f"Attempting to retrieve brands OCR results for video: {video_id}")
        brands_ocr = await s3_operations.get_brands_ocr_results(video_id)
        # app_logger.log_info(f"Successfully retrieved brands OCR results for video {video_id}")
        return brands_ocr

    except FileNotFoundError:
        app_logger.log_error(f"Brands OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Brands OCR results not found for video {video_id}")

    except Exception as e:
        app_logger.log_error(f"Error retrieving brands OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brands OCR results")
########################################################