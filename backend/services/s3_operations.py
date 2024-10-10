import uuid
import os
import json
import aiofiles
from typing import Optional, List, Dict, Set
from io import BytesIO
from fastapi import UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from core.config import settings
from core.aws import get_s3_client, multipart_upload
from core.state import set_upload_active, is_upload_active, remove_upload
from core.logging import logger
from utils.decorators import retry
from services import video_processing
from botocore.exceptions import ClientError
from models.detection_classes import DetectionResult, BrandInstance

## Uploads a video to S3
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
    logger.debug("Starting video chunk upload")
    set_upload_active(video_id)
    
    filename = 'original.mp4'
    s3_key = f'{video_id}/{filename}'

    async def perform_upload():
        try:
            chunk = await file.read()
            chunk_size = len(chunk)
            logger.debug(f"Read chunk of size {chunk_size} bytes")

            # Use a temporary file to store chunks
            temp_file_path = f"/tmp/{video_id}_temp.mp4"
            
            async with aiofiles.open(temp_file_path, 'ab') as f:
                await f.write(chunk)

            if chunk_number == total_chunks:
                # All chunks received, perform multipart upload
                file_size = os.path.getsize(temp_file_path)
                await multipart_upload(temp_file_path, settings.PROCESSING_BUCKET, s3_key, file_size)
                
                # Clean up temporary file
                os.remove(temp_file_path)
                
                logger.info("Video upload complete, starting video processing")
                remove_upload(video_id)
                background_tasks.add_task(video_processing.run_video_processing, video_id)
                logger.debug(f"Added video processing task for video_id: {video_id} to background tasks")
                return {"video_id": video_id, "status": "processing"}
            else:
                logger.debug("Chunk upload complete")
                return {
                    "video_id": video_id,
                    "status": "uploading",
                    "chunk": chunk_number,
                }

        except Exception as e:
            logger.error(f"Error during chunk upload", exc_info=True, extra={"error": str(e)})
            remove_upload(video_id)
            # Clean up temporary file in case of error
            if os.path.exists(f"/tmp/{video_id}_temp.mp4"):
                os.remove(f"/tmp/{video_id}_temp.mp4")
            raise

    return await perform_upload()
########################################################

## Cancels a video upload to s3
########################################################
async def cancel_video_upload(video_id: str):
    logger.debug(f"Attempting to cancel upload for video_id: {video_id}")

    if is_upload_active(video_id):
        set_upload_active(video_id, False)
        s3_key = f'{video_id}/original.mp4'

        try:
            logger.debug(f"Checking if object exists in S3 for video_id: {video_id}")
            async with get_s3_client() as s3_client:
                await s3_client.head_object(Bucket=settings.PROCESSING_BUCKET, Key=s3_key)

            logger.debug(f"Deleting object from S3 for video_id: {video_id}")
            await s3_client.delete_object(Bucket=settings.PROCESSING_BUCKET, Key=s3_key)

            logger.debug(f"Upload successfully cancelled for video_id: {video_id}")
            return {"status": "cancelled", "video_id": video_id}

        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"No file found to delete for cancelled upload: {video_id}")
            else:
                logger.error(f"Error deleting file for cancelled upload: {video_id}", exc_info=True)
            
            return {"status": "cancelled", "video_id": video_id}

    else:
        logger.error(f"Attempted to cancel non-active upload for video_id: {video_id}")
        return JSONResponse(status_code=404, content={"error": "Upload not found", "video_id": video_id})
########################################################

## Uploads processed reults to s3
########################################################
async def upload_processed_results(processed_dir, video_id, chunk_index):
    logger.debug(f"Starting upload of processed results for video_id: {video_id}, chunk_index: {chunk_index}")
    try:
        total_files = sum([len(files) for r, d, files in os.walk(processed_dir)])
        uploaded_files = 0
        total_bytes_uploaded = 0

        for root, _, files in os.walk(processed_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = f"{video_id}/chunks/{chunk_index}/{file}"
                
                file_size = os.path.getsize(local_path)
                logger.debug(f"Uploading file: {file}, size: {file_size} bytes")
                
                async with get_s3_client() as s3_client:
                    await s3_client.upload_file(
                        local_path, settings.PROCESSING_BUCKET, s3_key
                    )
                
                total_bytes_uploaded += file_size
                uploaded_files += 1
                
                logger.debug(f"Uploaded {uploaded_files}/{total_files} files")

        logger.debug(f"Completed upload of processed results for video_id: {video_id}, "
                            f"chunk_index: {chunk_index}. Total files: {uploaded_files}, "
                            f"Total bytes: {total_bytes_uploaded}")

    except Exception as e:
        logger.error(f"Error uploading processed results for video_id: {video_id}, "
                                f"chunk_index: {chunk_index}. Error: {str(e)}", exc_info=True)
        raise
########################################################

## Uploads a single video frame to s3
########################################################
@retry(exceptions=(ClientError,), tries=3, delay=1, backoff=2)
async def upload_frame_to_s3(key, body):
    try:
        logger.debug(f"Uploading frame to S3: {key}")
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=key,
                Body=body,
                ContentType='image/jpeg'
            )
        logger.debug(f"Successfully uploaded frame to S3: {key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload frame {key}: {str(e)}", exc_info=True)
        return False
########################################################

## Uploads batches of video frames to s3
########################################################
async def upload_frames_batch(bucket, frames):
    successful_uploads = 0
    total_size = 0
    try:
        for key, body in frames:
            logger.debug(f"Uploading frame to S3: {key}")
            async with get_s3_client() as s3_client:
                await s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/jpeg')
            total_size += len(body)
            successful_uploads += 1
        
        logger.debug(f"Successfully uploaded {successful_uploads} frames in batch")
    except Exception as e:
        logger.error(f"Failed to upload batch: {str(e)}", exc_info=True)
    
    return successful_uploads
########################################################


## Downloads a file for a given video ID and file type
########################################################
async def download_file_from_s3(video_id: str, file_type: str):
    logger.debug(f"Received download request for {file_type} of video: {video_id}")

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
        # app_logger.error(f"Invalid file type requested: {file_type} for video: {video_id}")
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        logger.debug(f"Checking if {file_type} exists in S3 for video: {video_id}")

        logger.debug(f"Generating pre-signed URL for {file_type} of video: {video_id}")
        async with get_s3_client() as s3_client:
            url = await s3_client.generate_presigned_url (
                'get_object',
                Params={'Bucket': settings.PROCESSING_BUCKET, 'Key': key},
                ExpiresIn=3600,
                HttpMethod='GET'
            )

        logger.debug(f"Successfully generated download URL for {file_type} of video: {video_id}")
        return RedirectResponse(url=url)

    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"{file_type.capitalize()} not found for video: {video_id}")
            raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} not found")
        else:
            logger.error(f"Error downloading {file_type} for video {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error downloading {file_type}")
########################################################

## Load raw video OCR results for post-processing
########################################################
async def load_ocr_results(video_id: str) -> List[Dict]:
    logger.debug(f"Video Processing - Brand Detection - Step 1.2: Loading OCR results for video: {video_id}")

    key = f'{video_id}/ocr/raw_ocr.json'
    try:
        logger.debug(f"Attempting to retrieve OCR results from S3 for video: {video_id}")
        
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=key
            )
        
        # Read the data in chunks asynchronously
        data = await response['Body'].read()
        
        # Parse JSON synchronously
        try:
            ocr_results = json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Invalid OCR results format for video: {video_id}")
        
        logger.debug(f"Successfully loaded OCR results for video: {video_id}. Size: {len(ocr_results)} frames")
        return ocr_results
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"OCR results not found for video: {video_id}")
            raise FileNotFoundError(f"OCR results not found for video: {video_id}")
        else:
            logger.error(f"Error loading OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise
########################################################

## Save processed OCR results for video
########################################################
async def save_processed_ocr_results(video_id: str, cleaned_results: List[Dict]):
    logger.debug(f"Saving processed OCR results for video: {video_id}")

    key = f'{video_id}/ocr/processed_ocr.json'

    try:
        processed_data = json.dumps(cleaned_results, indent=2)
        data_size = len(processed_data)

        logger.debug(f"Attempting to save processed OCR results to S3 for video: {video_id}")
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=key,
                Body=processed_data,
                ContentType='application/json'
            )
        
        logger.debug(f"Successfully saved processed OCR results for video: {video_id}. Size: {data_size} bytes")

    except ClientError as e:
        logger.error(f"Error saving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## Save brands OCR results for video
########################################################
async def save_brands_ocr_results(video_id: str, brand_results: List[DetectionResult]):
    logger.debug(f"Video Processing - Brand Detection - Step 3.5: Saving brands OCR results for video: {video_id}")
    key = f'{video_id}/ocr/brands_ocr.json'
    try:
        # Convert DetectionResult objects to serializable dictionaries
        serializable_results = [
            {
                "frame_number": result.frame_number,
                "detected_brands": [brand.to_dict() for brand in result.detected_brands]
            }
            for result in brand_results
        ]
        brand_data = json.dumps(serializable_results, indent=2)
        data_size = len(brand_data)
        logger.debug(f"Video Processing - Brand Detection - Step 3.5: Attempting to save brands OCR results to S3 for video: {video_id}")
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=key,
                Body=brand_data,
                ContentType='application/json'
            )
        logger.debug(f"Video Processing - Brand Detection - Step 3.5: Successfully saved brands OCR results for video: {video_id}. Size: {data_size} bytes")
    except ClientError as e:
        logger.error(f"Video Processing - Brand Detection - Step 3.5: Error saving brands OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Video Processing - Brand Detection - Step 3.5: Unexpected error saving brands OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## Create and save brands OCR results table for video
########################################################
async def create_and_save_brand_table(video_id: str, brand_stats: Dict[str, Dict]):
    logger.debug(f"Video Processing - Brand Detection - Step 3.6: Creating and saving detected brand table for video: {video_id}")

    try:
        brand_table_data = json.dumps(brand_stats, indent=2)
        data_size = len(brand_table_data)
        logger.debug(f"Video Processing - Brand Detection - Step 3.6: Attempting to save detected brand table to S3 for video: {video_id}")
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/brands_table.json',
                Body=brand_table_data,
                ContentType='application/json'
            )
        logger.debug(f"Video Processing - Brand Detection - Step 3.6: Successfully saved detected brand table for video: {video_id}. Size: {data_size} bytes")
        return brand_stats
    except ClientError as e:
        logger.error(f"Video Processing - Brand Detection - Step 3.6: Error saving detected brand table for video {video_id}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Video Processing - Brand Detection - Step 3.6: Unexpected error saving brand table for video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## Get video's wordcloud
########################################################
async def get_word_cloud(video_id: str):
    logger.debug(f"Received request for word cloud of video: {video_id}")
    wordcloud_key = f'{video_id}/ocr/wordcloud.jpg'

    try:
        logger.debug(f"Attempting to retrieve word cloud from S3 for video: {video_id}")

        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=wordcloud_key
            )
        image_data = await response['Body'].read()
        image_size = len(image_data)
        logger.debug(f"Successfully retrieved word cloud for video {video_id}. Size: {image_size} bytes")

        return StreamingResponse(BytesIO(image_data), media_type="image/jpeg")

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Word cloud not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Word cloud not found for video {video_id}")

    except Exception as e:
        logger.error(f"Error retrieving word cloud for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## Get brands OCR table for video
########################################################
async def get_brands_ocr_table(video_id: str):
    logger.debug(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_table.json'

    try:
        logger.debug(f"Attempting to retrieve brand OCR table from S3 for video: {video_id}")
        
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=ocr_key
            )
        data = await response['Body'].read()
        data_size = len(data)

        brands_table = json.loads(data.decode('utf-8'))
        logger.debug(f"Successfully retrieved brand OCR table for video {video_id}. Size: {data_size} bytes")
        
        return brands_table

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Brands OCR table not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Brands OCR table not found for video {video_id}")
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for brand OCR table of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing brand OCR table")

    except s3_client.exceptions.ClientError as e:
        logger.error(f"S3 client error retrieving brand OCR table for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR table from storage")

    except Exception as e:
        logger.error(f"Unexpected error retrieving brand OCR table for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving brand OCR table")
########################################################

## Get video's OCR results
########################################################
async def get_brands_ocr_results(video_id: str):
    logger.debug(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_ocr.json'
    
    try:
        logger.debug(f"Attempting to retrieve brand OCR results from S3 for video: {video_id}")

        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=ocr_key
            )
        data = await response['Body'].read()
        data_size = len(data)

        ocr_results = json.loads(data.decode('utf-8'))
        logger.debug(f"Successfully retrieved brand OCR results for video {video_id}. Size: {data_size} bytes")
        
        # Convert the JSON data back to DetectionResult objects
        detection_results = [
            DetectionResult(
                frame_number=frame_data['frame_number'],
                detected_brands=[
                    BrandInstance(**brand_data)
                    for brand_data in frame_data['detected_brands']
                ]
            )
            for frame_data in ocr_results
        ]
        
        return detection_results

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Brand OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Brand OCR results not found")
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for brand OCR results of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing brand OCR results")

    except s3_client.exceptions.ClientError as e:
        logger.error(f"S3 client error retrieving brand OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR results from storage")

    except Exception as e:
        logger.error(f"Unexpected error retrieving brand OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving brand OCR results")
########################################################

## Get video's processed OCR results
########################################################
async def get_processed_ocr_results(video_id: str):
    logger.debug(f"Received request for processed OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/processed_ocr.json'

    try:
        logger.debug(f"Attempting to retrieve processed OCR results from S3 for video: {video_id}")

        async with get_s3_client() as s3_client:
            response = s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=ocr_key
            )
        data = await response['Body'].read()
        data_size = len(data)

        ocr_results = json.loads(data.decode('utf-8'))
        logger.debug(f"Successfully retrieved processed OCR results for video {video_id}. Size: {data_size} bytes")
        
        return ocr_results

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Processed OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Processed OCR results not found for video {video_id}")
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for processed OCR results of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing OCR results")

    except s3_client.exceptions.ClientError as e:
        logger.error(f"S3 client error retrieving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results from storage")

    except Exception as e:
        logger.error(f"Unexpected error retrieving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving processed OCR results")
########################################################

## Get video processed OCR results
########################################################
async def get_brands_ocr_results(video_id: str):
    logger.debug(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_ocr.json'
    
    try:
        logger.debug(f"Attempting to retrieve brand OCR results from S3 for video: {video_id}")

        async with get_s3_client() as s3_client:
            response = s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=ocr_key
            )
        data = await response['Body'].read()
        data_size = len(data)

        ocr_results = json.loads(data.decode('utf-8'))
        logger.debug(f"Successfully retrieved brand OCR results for video {video_id}. Size: {data_size} bytes")
        
        return ocr_results

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Brand OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Brand OCR results not found")
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for brand OCR results of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing brand OCR results")

    except s3_client.exceptions.ClientError as e:
        logger.error(f"S3 client error retrieving brand OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR results from storage")

    except Exception as e:
        logger.error(f"Unexpected error retrieving brand OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving brand OCR results")
########################################################

## Get processed video for a given video ID
########################################################
async def get_processed_video(video_id: str) -> Optional[str]:
    logger.debug(f"Attempting to generate pre-signed URL for processed video: {video_id}")
    key = f'{video_id}/processed_video.mp4'
    
    async with get_s3_client() as s3_client:
        try:
            # Check if the file exists
            await s3_client.head_object(Bucket=settings.PROCESSING_BUCKET, Key=key)
            
            # Generate pre-signed URL
            url = await s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.PROCESSING_BUCKET,
                    'Key': key
                },
                ExpiresIn=3600  # URL expires in 1 hour
            )
            logger.debug(f"Successfully generated pre-signed URL for video {video_id}")
            return url
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey' or error_code == '404':
                logger.warning(f"Processed video not found for {video_id}")
            else:
                logger.error(f"Error accessing or generating URL for video {video_id}: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error for video {video_id}: {str(e)}", exc_info=True)
            return None
########################################################