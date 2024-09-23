import json
from fastapi import APIRouter, HTTPException
from core.config import settings
from core.logging import video_logger, AppLogger
from core.aws import get_s3_client
from services import s3_operations
from services.ocr_processing import main_ocr_processing
from utils import utils

# Create a global instance of AppLogger
app_logger = AppLogger()

router = APIRouter()

## REPROCESS OCR ENDPOINT (POST)
## Reprocesses the raw OCR data for the video
########################################################
@router.post("/{video_id}/ocr/reprocess-ocr")
async def reprocess_ocr(video_id: str):
    with video_logger("api-endpoints", is_api_log=True) as vlogger:
        @vlogger.log_performance
        async def _reprocess_ocr():
            vlogger.logger.info(f"Received request to reprocess OCR for video: {video_id}")

            try:
                vlogger.logger.debug(f"Fetching processing stats for video: {video_id}")
                s3_client = await get_s3_client()
                stats_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/processing_stats.json')
                stats = json.loads(stats_obj['Body'].read().decode('utf-8'))
                fps = stats['video']['video_fps']

                vlogger.logger.debug(f"Retrieved FPS: {fps} for video: {video_id}")
                video_resolution = await vlogger.log_performance(utils.get_video_resolution)(vlogger, video_id)

                vlogger.logger.debug(f"Starting OCR reprocessing for video: {video_id} with resolution: {video_resolution}")
                result = await vlogger.log_performance(main_ocr_processing.post_process_ocr)(vlogger, video_id, fps, video_resolution, s3_client)

                vlogger.logger.info(f"OCR reprocessing completed for video {video_id}, identified {len(result)} brands")
                return {"status": "success", "message": f"OCR reprocessing completed, identified {len(result)} brands"}

            except Exception as e:
                vlogger.logger.error(f"Error reprocessing OCR for video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error reprocessing OCR")

        return await _reprocess_ocr()
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