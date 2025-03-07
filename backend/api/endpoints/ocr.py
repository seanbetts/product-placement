import traceback
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from core.logging import logger
from services import s3_operations
from services.ocr_processing import brand_detection

router = APIRouter()

## REPROCESS OCR ENDPOINT (POST)
## Reprocesses the raw OCR data for the video
########################################################
@router.post("/{video_id}/ocr/reprocess-ocr")
async def reprocess_ocr(video_id: str) -> Dict[str, Any]:
    logger.info(f"Received request to reprocess OCR for video {video_id}")
    
    status_tracker = StatusTracker(video_id)
    await status_tracker.update_s3_status()

    video_details = await VideoDetails.create(video_id)
    
    try:
        logger.info(f"Starting to reprocess OCR data for video: {video_id}")
        result = await brand_detection.detect_brands(video_id, status_tracker, video_details)
        logger.info(f"Reprocessing of OCR Data for video {video_id} complete, identified {len(result)} brands")
        
        return {"status": "success", "message": f"OCR data reprocessing completed for video {video_id}"}
    except Exception as e:
        logger.error(f"Error reprocessing OCR data for video {video_id}: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error reprocessing OCR data")
########################################################

## WORDCLOUD ENDPOINT (GET)
## Returns the wordcloud for a video
########################################################
@router.get("/{video_id}/ocr/wordcloud")
async def get_word_cloud(video_id: str):
    logger.debug(f"Received request for word cloud of video: {video_id}")
    try:
        logger.debug(f"Attempting to retrieve word cloud for video: {video_id}")
        wordcloud = await s3_operations.get_word_cloud(video_id)
        logger.debug(f"Successfully retrieved word cloud for video {video_id}")
        return wordcloud

    except FileNotFoundError:
        logger.error(f"Word cloud not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Word cloud not found")

    except Exception as e:
        logger.error(f"Error retrieving word cloud for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## BRANDS OCR TABLE ENDPOINT (GET)
## Returns the brands OCR table for a video
########################################################
@router.get("/{video_id}/ocr/brands-ocr-table")
async def get_brands_ocr_table(video_id: str):
    logger.debug(f"Received request for brand OCR table of video: {video_id}")
    try:
        logger.debug(f"Attempting to retrieve brand OCR table for video: {video_id}")
        table = await s3_operations.get_brands_ocr_table(video_id)
        logger.debug(f"Successfully retrieved brand OCR table for video {video_id}")
        return table

    except FileNotFoundError:
        logger.error(f"Brands OCR table not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Brands OCR table not found")

    except Exception as e:
        logger.error(f"Error retrieving brands OCR table for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brands OCR table")
########################################################

## OCR RESULTS ENDPOINT (GET)
## Returns the OCR results for a video
########################################################
@router.get("/{video_id}/ocr/results")
async def get_ocr_results(video_id: str):
    logger.debug(f"Received request for OCR results of video: {video_id}")
    try:
        logger.debug(f"Attempting to retrieve OCR results for video: {video_id}")
        results = await s3_operations.get_ocr_results(video_id)
        logger.debug(f"Successfully retrieved OCR results for video {video_id}")
        return results

    except FileNotFoundError:
        logger.error(f"OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail="OCR results not found")

    except Exception as e:
        logger.error(f"Error retrieving OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving OCR results")
########################################################

## PROCESSED OCR DATA ENDPOINT (GET)
## Returns the processed OCR results for a video
########################################################
@router.get("/{video_id}/ocr/processed-ocr")
async def get_processed_ocr(video_id: str):
    logger.debug(f"Received request for processed OCR results for video: {video_id}")
    try:
        logger.debug(f"Attempting to retrieve processed OCR results for video: {video_id}")
        processed_ocr = await s3_operations.get_processed_ocr_results(video_id)
        logger.debug(f"Successfully retrieved processed OCR results for video {video_id}")
        return processed_ocr

    except FileNotFoundError:
        logger.error(f"Processed OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Processed OCR results not found for video {video_id}")

    except Exception as e:
        logger.error(f"Error retrieving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results")
########################################################

## BRANDS OCR DATA ENDPOINT (GET)
## Returns the brands OCR results for a video
########################################################
@router.get("/{video_id}/ocr/brands-ocr")
async def get_brands_ocr(video_id: str):
    logger.debug(f"Received request for brands OCR results for video: {video_id}")
    try:
        logger.debug(f"Attempting to retrieve brands OCR results for video: {video_id}")
        brands_ocr = await s3_operations.get_brands_ocr_results(video_id)
        logger.debug(f"Successfully retrieved brands OCR results for video {video_id}")
        return brands_ocr

    except FileNotFoundError:
        logger.error(f"Brands OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Brands OCR results not found for video {video_id}")

    except Exception as e:
        logger.error(f"Error retrieving brands OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brands OCR results")
########################################################