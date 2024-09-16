import uuid
import asyncio
import os
import json
import tempfile
from typing import Optional, List, Dict, Set
from io import BytesIO
from fastapi import File, UploadFile, BackgroundTasks, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client
from core.state import set_upload_active, is_upload_active, remove_upload
from services import video_processing
from botocore.exceptions import ClientError

s3_client = get_s3_client()

## Uploads a video to s3
########################################################
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    chunk_number: int,
    total_chunks: int,
    video_id: Optional[str] = None
):
    if not video_id:
        video_id = str(uuid.uuid4())
    
    log_context = {
        "video_id": video_id,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks,
        "upload_filename": file.filename,
        "content_type": file.content_type,
    }
    
    set_upload_active(video_id)
    s3_key = f'{video_id}/original.mp4'
    s3_client = get_s3_client()

    try:
        chunk = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name
            
            if chunk_number > 1:
                await asyncio.to_thread(s3_client.download_file, settings.PROCESSING_BUCKET, s3_key, temp_filename)
            
            with open(temp_filename, 'ab') as f:
                f.write(chunk)
            
            if not is_upload_active(video_id):
                raise Exception("Upload cancelled")

            await asyncio.to_thread(s3_client.upload_file, temp_filename, settings.PROCESSING_BUCKET, s3_key)
        
        os.unlink(temp_filename)
        
        if chunk_number == total_chunks:
            remove_upload(video_id)
            background_tasks.add_task(video_processing.run_video_processing, video_id)
            return {"video_id": video_id, "status": "processing"}
        else:
            return {
                "video_id": video_id,
                "status": "uploading",
                "chunk": chunk_number,
            }

    except Exception as e:
        logger.error("Error during chunk upload", exc_info=True, extra={**log_context, "error": str(e)})
        remove_upload(video_id)
        raise
########################################################

## Cancels a video upload to s3
########################################################
async def cancel_video_upload(video_id: str):
    if is_upload_active(video_id):
        # active_uploads[video_id] = False
        set_upload_active(video_id, False)
        s3_key = f'{video_id}/original.mp4'
        try:
            # Check if the object exists
            s3_client.head_object(Bucket=settings.PROCESSING_BUCKET, Key=s3_key)
            # If it exists, delete it
            s3_client.delete_object(Bucket=settings.PROCESSING_BUCKET, Key=s3_key)
            logger.info(f"Upload cancelled for video_id: {video_id}")
            return {"status": "cancelled", "video_id": video_id}
        except s3_client.exceptions.ClientError as e:
            # If the object was not found, it's okay, just log it
            if e.response['Error']['Code'] == '404':
                logger.info(f"No file found to delete for cancelled upload: {video_id}")
            else:
                logger.error(f"Error deleting file for cancelled upload: {video_id}", exc_info=True)
        return {"status": "cancelled", "video_id": video_id}
    else:
        return JSONResponse(status_code=404, content={"error": "Upload not found", "video_id": video_id})
########################################################

## Uploads a video frame to s3
########################################################
def upload_frame_to_s3(s3_client, bucket, key, body):
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/jpeg')
        return True
    except Exception as e:
        logger.error(f"Failed to upload frame {key}: {str(e)}")
        return False
########################################################

## Downloads a file for a given video ID and file type
########################################################
async def download_file_from_s3(video_id: str, file_type: str):
    logger.info(f"Received download request for {file_type} of video: {video_id}")

    if file_type == "video":
        key = f'{video_id}/original.mp4'
        filename = f"{video_id}_video.mp4"
    elif file_type == "audio":
        key = f'{video_id}/audio.mp3'
        filename = f"{video_id}_audio.mp3"
    elif file_type == "transcript":
        key = f'{video_id}/transcripts/transcript.txt'
        filename = f"{video_id}_transcript.txt"
    elif file_type == "word-cloud":
        key = f'{video_id}/ocr/wordcloud.jpg'
        filename = f"{video_id}_wordcloud.jpg"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Check if the object exists
        s3_client.head_object(Bucket=settings.PROCESSING_BUCKET, Key=key)
        
        # Create a pre-signed URL for the object
        url = s3_client.generate_presigned_url('get_object',
                                               Params={'Bucket': settings.PROCESSING_BUCKET, 'Key': key},
                                               ExpiresIn=3600,
                                               HttpMethod='GET')
        
        # Redirect to the pre-signed URL
        return RedirectResponse(url=url)
    
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} not found")
        else:
            logger.error(f"Error downloading {file_type} for video {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading {file_type}")
########################################################

## Load raw video OCR results for post-processing
########################################################
def load_ocr_results(s3_client, video_id: str) -> List[Dict]:
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/ocr/ocr_results.json')
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise FileNotFoundError(f"OCR results not found for video: {video_id}")
        else:
            raise
########################################################

## Save processed OCR results for video
########################################################
async def save_processed_ocr_results(s3_client, video_id: str, cleaned_results: List[Dict]):
    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=settings.PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/processed_ocr.json',
            Body=json.dumps(cleaned_results, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Saved processed OCR results for video: {video_id}")
    except ClientError as e:
        logger.error(f"Error saving processed OCR results for video {video_id}: {str(e)}")
        raise
########################################################

## Save brands OCR results for video
########################################################
async def save_brands_ocr_results(s3_client, video_id: str, brand_results: List[Dict]):
    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=settings.PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/brands_ocr.json',
            Body=json.dumps(brand_results, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Saved brands OCR results for video: {video_id}")
    except ClientError as e:
        logger.error(f"Error saving brands OCR results for video {video_id}: {str(e)}")
        raise
########################################################

## Create and save brands OCR results table for video
########################################################
def create_and_save_brand_table(s3_client, video_id: str, brand_appearances: Dict[str, Set[int]], fps: float):
    brand_stats = {}
    min_frames = int(fps)  # Minimum number of frames (1 second)

    for brand, frames in brand_appearances.items():
        frame_list = sorted(frames)
        if len(frame_list) >= min_frames:
            brand_stats[brand] = {
                "frame_count": len(frame_list),
                "time_on_screen": round(len(frame_list) / fps, 2),
                "first_appearance": frame_list[0],
                "last_appearance": frame_list[-1]
            }
        else:
            logger.info(f"Discarded brand '{brand}' as it appeared for less than 1 second ({len(frame_list)} frames)")

    try:
        s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/brands_table.json',
            Body=json.dumps(brand_stats, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Brand table created and saved for video: {video_id}")
        return brand_stats
    except ClientError as e:
        logger.error(f"Error saving brand table for video {video_id}: {str(e)}")
        raise
########################################################


## Get video's wordcloud
########################################################
async def get_word_cloud(video_id: str):
    logger.info(f"Received request for word cloud of video: {video_id}")
    wordcloud_key = f'{video_id}/ocr/wordcloud.jpg'
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=wordcloud_key)
        image_data = response['Body'].read()
        return StreamingResponse(BytesIO(image_data), media_type="image/jpeg")
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Word cloud not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving word cloud for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## Get brands OCR table for video
########################################################
async def get_brands_ocr_table(video_id: str):
    logger.info(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_table.json'
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=ocr_key)
        brands_table = json.loads(response['Body'].read().decode('utf-8'))
        return brands_table
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Brands OCR table not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving brand OCR table for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR table")
########################################################

## Get video's OCR results
########################################################
async def get_ocr_results(video_id: str):
    logger.info(f"Received request for OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/ocr_results.json'
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"OCR results not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving OCR results")
########################################################

## Get video's processed OCR results
########################################################
async def get_processed_ocr_results(video_id: str):
    logger.info(f"Received request for processed OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/processed_ocr.json'
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Processed OCR results not found for video {video_id}")
    except Exception as e:
        logger.error(f"Error retrieving processed OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results")
########################################################

## Get video processed OCR results
########################################################
async def get_brands_ocr_results(video_id: str):
    logger.info(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_ocr.json'
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=ocr_key)
        ocr_results = json.loads(response['Body'].read().decode('utf-8'))
        return ocr_results
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Brands OCR results not found")
    except Exception as e:
        logger.error(f"Error retrieving brand OCR results for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR results")
########################################################