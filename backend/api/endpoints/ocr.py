import json
from fastapi import APIRouter, HTTPException
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client
from services import s3_operations
from services.ocr_processing import main_ocr_processing
from utils import utils

router = APIRouter()

## REPROCESS OCR ENDPOINT (POST)
## Reprocesses the raw OCR data for the video
########################################################
@router.post("/{video_id}/ocr/reprocess-ocr")
async def reprocess_ocr(video_id: str):
    logger.info(f"Received request to reprocess OCR for video: {video_id}")
    try:
        s3_client = get_s3_client()
        stats_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/processing_stats.json')
        stats = json.loads(stats_obj['Body'].read().decode('utf-8'))
        fps = stats['video']['video_fps']
        video_resolution = await utils.get_video_resolution(video_id)
        result = await main_ocr_processing.post_process_ocr(video_id, fps, video_resolution, s3_client)
        return {"status": "success", "message": f"OCR reprocessing completed, identified {len(result)} brands"}
    except Exception as e:
        logger.error(f"Error reprocessing OCR for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reprocessing OCR")
########################################################

## WORDCLOUD ENDPOINT (GET)
## Returns the wordcloud for a video
########################################################
@router.get("/{video_id}/ocr/wordcloud")
async def get_word_cloud(video_id: str):
    logger.info(f"Received request for word cloud of video: {video_id}")
    try:
        wordcloud = await s3_operations.get_word_cloud(video_id)
        return wordcloud
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Word cloud not found")
    except Exception as e:
        logger.error(f"Error retrieving word cloud for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## BRANDS OCR TABLE ENDPOINT (GET)
## Returns the brands OCR table for a video
########################################################
@router.get("/{video_id}/ocr/brands-ocr-table")
async def get_brands_ocr_table(video_id: str):
    logger.info(f"Received request for brand OCR table of video: {video_id}")
    try:
        table = await s3_operations.get_brands_ocr_table(video_id)
        return table
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Brands OCR table not found")
    except Exception as e:
        logger.error(f"Error retrieving brands OCR table for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving brands OCR table")
########################################################

## OCR RESULTS ENDPOINT (GET)
## Returns the OCR results for a video
########################################################
@router.get("/{video_id}/ocr/results")
async def get_ocr_results(video_id: str):
    logger.info(f"Received request for OCR results of video: {video_id}")
    try:
        results = await s3_operations.get_ocr_results(video_id)
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="OCR results not found")
    except Exception as e:
        logger.error(f"Error retrieving OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving OCR results")
########################################################

## PROCESSED OCR DATA ENDPOINT (GET)
## Returns the processed OCR results for a video
########################################################
@router.get("/{video_id}/ocr/processed-ocr")
async def get_processed_ocr(video_id: str):
    logger.info(f"Received request for processed OCR results for video: {video_id}")
    try:
        processed_ocr = await s3_operations.get_processed_ocr_results(video_id)
        return processed_ocr
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Processed OCR results not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving processed OCR for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results")
########################################################

## BRANDS OCR DATA ENDPOINT (GET)
## Returns the brands OCR results for a video
########################################################
@router.get("/{video_id}/ocr/brands-ocr")
async def get_brands_ocr(video_id: str):
    logger.info(f"Received request for brands OCR resutls for video: {video_id}")
    try:
        brands_ocr = await s3_operations.get_brands_ocr_results(video_id)
        return brands_ocr
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Brands OCR results not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving brands OCR for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving brands OCR results")
########################################################