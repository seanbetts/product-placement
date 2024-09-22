import uuid
import asyncio
import os
import json
import tempfile
from typing import Optional, List, Dict, Set
from io import BytesIO
from fastapi import UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from core.config import settings
from core.aws import get_s3_client
from core.state import set_upload_active, is_upload_active, remove_upload
from core.logging import AppLogger
from services import video_processing
from botocore.exceptions import ClientError

# Create a global instance of AppLogger
app_logger = AppLogger()

# Create a global instance of s3_client
s3_client = get_s3_client()

## Uploads a video to S3
########################################################
async def upload_video(
    vlogger,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    chunk_number: int,
    total_chunks: int,
    video_id: Optional[str] = None
):
    # Set video_id at the beginning
    if not video_id:
        video_id = str(uuid.uuid4())

    log_context = {
        "video_id": video_id,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks,
        "upload_filename": file.filename,
        "content_type": file.content_type,
    }
    
    vlogger.logger.info("Starting video chunk upload", extra=log_context)
    set_upload_active(video_id)
    s3_key = f'{video_id}/original.mp4'
    s3_client = get_s3_client()

    @vlogger.log_performance
    async def perform_upload():
        try:
            chunk = await file.read()
            chunk_size = len(chunk)
            vlogger.logger.debug(f"Read chunk of size {chunk_size} bytes", extra=log_context)

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name
                if chunk_number > 1:
                    vlogger.logger.debug("Downloading existing file for appending", extra=log_context)
                    await asyncio.to_thread(
                        s3_client.download_file, settings.PROCESSING_BUCKET, s3_key, temp_filename
                    )
                    vlogger.log_s3_operation("download", os.path.getsize(temp_filename))
                
                with open(temp_filename, 'ab') as f:
                    f.write(chunk)

            if not is_upload_active(video_id):
                raise Exception("Upload cancelled")

            vlogger.logger.debug("Uploading chunk to S3", extra=log_context)
            await asyncio.to_thread(
                s3_client.upload_file, temp_filename, settings.PROCESSING_BUCKET, s3_key
            )
            vlogger.log_s3_operation("upload", os.path.getsize(temp_filename))
            os.unlink(temp_filename)

            if chunk_number == total_chunks:
                vlogger.logger.info("Upload complete, starting video processing", extra=log_context)
                remove_upload(video_id)
                background_tasks.add_task(video_processing.run_video_processing, vlogger, video_id)
                return {"video_id": video_id, "status": "processing"}
            else:
                vlogger.logger.info("Chunk upload complete", extra=log_context)
                return {
                    "video_id": video_id,
                    "status": "uploading",
                    "chunk": chunk_number,
                }
        except Exception as e:
            vlogger.logger.error("Error during chunk upload", exc_info=True, extra={**log_context, "error": str(e)})
            remove_upload(video_id)
            raise

    return await perform_upload()
########################################################

## Cancels a video upload to s3
########################################################
async def cancel_video_upload(vlogger, video_id: str):
    @vlogger.log_performance
    async def _cancel_video_upload():
        vlogger.logger.info(f"Attempting to cancel upload for video_id: {video_id}")

        if is_upload_active(video_id):
            set_upload_active(video_id, False)
            s3_key = f'{video_id}/original.mp4'

            try:
                vlogger.logger.debug(f"Checking if object exists in S3 for video_id: {video_id}")
                await vlogger.log_performance(s3_client.head_object)(Bucket=settings.PROCESSING_BUCKET, Key=s3_key)

                vlogger.logger.info(f"Deleting object from S3 for video_id: {video_id}")
                await vlogger.log_performance(s3_client.delete_object)(Bucket=settings.PROCESSING_BUCKET, Key=s3_key)
                vlogger.log_s3_operation("delete", 0)  # Log the delete operation

                vlogger.logger.info(f"Upload successfully cancelled for video_id: {video_id}")
                return {"status": "cancelled", "video_id": video_id}

            except s3_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    vlogger.logger.info(f"No file found to delete for cancelled upload: {video_id}")
                else:
                    vlogger.logger.error(f"Error deleting file for cancelled upload: {video_id}", exc_info=True)
                
                return {"status": "cancelled", "video_id": video_id}

        else:
            vlogger.logger.warning(f"Attempted to cancel non-active upload for video_id: {video_id}")
            return JSONResponse(status_code=404, content={"error": "Upload not found", "video_id": video_id})

    return await _cancel_video_upload()
########################################################

## Uploads processed reults to s3
########################################################
async def upload_processed_results(vlogger, processed_dir, video_id, s3_client, chunk_index):
    @vlogger.log_performance
    async def _upload_processed_results():
        vlogger.logger.info(f"Starting upload of processed results for video_id: {video_id}, chunk_index: {chunk_index}")
        
        try:
            total_files = sum([len(files) for r, d, files in os.walk(processed_dir)])
            uploaded_files = 0
            total_bytes_uploaded = 0

            for root, _, files in os.walk(processed_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_key = f"{video_id}/chunks/{chunk_index}/{file}"
                    
                    file_size = os.path.getsize(local_path)
                    vlogger.logger.debug(f"Uploading file: {file}, size: {file_size} bytes")
                    
                    await vlogger.log_performance(s3_client.upload_file)(
                        local_path, settings.PROCESSING_BUCKET, s3_key
                    )
                    
                    vlogger.log_s3_operation("upload", file_size)
                    total_bytes_uploaded += file_size
                    uploaded_files += 1
                    
                    vlogger.logger.debug(f"Uploaded {uploaded_files}/{total_files} files")

            vlogger.logger.info(f"Completed upload of processed results for video_id: {video_id}, "
                                f"chunk_index: {chunk_index}. Total files: {uploaded_files}, "
                                f"Total bytes: {total_bytes_uploaded}")

        except Exception as e:
            vlogger.logger.error(f"Error uploading processed results for video_id: {video_id}, "
                                 f"chunk_index: {chunk_index}. Error: {str(e)}", exc_info=True)
            raise

    return await _upload_processed_results()
########################################################

## Uploads a single video frame to s3
########################################################
def upload_frame_to_s3(vlogger, s3_client, bucket, key, body):
    @vlogger.log_performance
    def _upload_frame_to_s3():
        try:
            vlogger.logger.debug(f"Uploading frame to S3: {key}")
            
            s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/jpeg')
            
            vlogger.log_s3_operation("upload", len(body))
            vlogger.logger.info(f"Successfully uploaded frame to S3: {key}")
            
            return True
        except Exception as e:
            vlogger.logger.error(f"Failed to upload frame {key}: {str(e)}", exc_info=True)
            return False

    return _upload_frame_to_s3()
########################################################

## Uploads batches of video frames to s3
########################################################
async def upload_frames_batch(vlogger, s3_client, bucket, frames):
    successful_uploads = 0
    total_size = 0
    try:
        for key, body in frames:
            vlogger.logger.debug(f"Uploading frame to S3: {key}")
            await asyncio.to_thread(s3_client.put_object, Bucket=bucket, Key=key, Body=body, ContentType='image/jpeg')
            total_size += len(body)
            successful_uploads += 1
        
        vlogger.log_s3_operation("upload", total_size)
        vlogger.logger.info(f"Successfully uploaded {successful_uploads} frames in batch")
    except Exception as e:
        vlogger.logger.error(f"Failed to upload batch: {str(e)}", exc_info=True)
    
    return successful_uploads
########################################################


## Downloads a file for a given video ID and file type
########################################################
async def download_file_from_s3(video_id: str, file_type: str):
    # app_logger.log_info(f"Received download request for {file_type} of video: {video_id}")

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
        # app_logger.log_info(f"Checking if {file_type} exists in S3 for video: {video_id}")

        # app_logger.log_info(f"Generating pre-signed URL for {file_type} of video: {video_id}")
        url = await asyncio.to_thread (
            s3_client.generate_presigned_url,
            'get_object',
            Params={'Bucket': settings.PROCESSING_BUCKET, 'Key': key},
            ExpiresIn=3600,
            HttpMethod='GET'
        )

        # app_logger.log_info(f"Successfully generated download URL for {file_type} of video: {video_id}")
        return RedirectResponse(url=url)

    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            app_logger.log_error(f"{file_type.capitalize()} not found for video: {video_id}")
            raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} not found")
        else:
            app_logger.log_error(f"Error downloading {file_type} for video {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error downloading {file_type}")
########################################################

## Load raw video OCR results for post-processing
########################################################
async def load_ocr_results(vlogger, s3_client, video_id: str) -> List[Dict]:
    @vlogger.log_performance
    async def _load_ocr_results():
        vlogger.logger.info(f"Loading OCR results for video: {video_id}")
        key = f'{video_id}/ocr/raw_ocr.json'
        try:
            vlogger.logger.debug(f"Attempting to retrieve OCR results from S3 for video: {video_id}")
            
            # Use asyncio.to_thread for the S3 operation
            response = await asyncio.to_thread(
                s3_client.get_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=key
            )
            
            # Read the data in chunks asynchronously
            data = await asyncio.to_thread(response['Body'].read)
            vlogger.log_s3_operation("download", len(data))
            
            # Parse JSON asynchronously
            ocr_results = await asyncio.to_thread(json.loads, data.decode('utf-8'))
            
            vlogger.logger.info(f"Successfully loaded OCR results for video: {video_id}. Size: {len(ocr_results)} frames")
            return ocr_results
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                vlogger.logger.warning(f"OCR results not found for video: {video_id}")
                raise FileNotFoundError(f"OCR results not found for video: {video_id}")
            else:
                vlogger.logger.error(f"Error loading OCR results for video {video_id}: {str(e)}", exc_info=True)
                raise
        except json.JSONDecodeError as e:
            vlogger.logger.error(f"Error decoding OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise ValueError(f"Invalid OCR results format for video: {video_id}")

    return await _load_ocr_results()
########################################################

## Save processed OCR results for video
########################################################
async def save_processed_ocr_results(vlogger, s3_client, video_id: str, cleaned_results: List[Dict]):
    @vlogger.log_performance
    async def _save_processed_ocr_results():
        vlogger.logger.info(f"Saving processed OCR results for video: {video_id}")
        key = f'{video_id}/ocr/processed_ocr.json'

        try:
            processed_data = json.dumps(cleaned_results, indent=2)
            data_size = len(processed_data)

            vlogger.logger.debug(f"Attempting to save processed OCR results to S3 for video: {video_id}")
            await vlogger.log_performance(asyncio.to_thread)(
                s3_client.put_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=key,
                Body=processed_data,
                ContentType='application/json'
            )
            
            vlogger.log_s3_operation("upload", data_size)
            vlogger.logger.info(f"Successfully saved processed OCR results for video: {video_id}. Size: {data_size} bytes")

        except ClientError as e:
            vlogger.logger.error(f"Error saving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            vlogger.logger.error(f"Unexpected error saving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise

    return await _save_processed_ocr_results()
########################################################

## Save brands OCR results for video
########################################################
async def save_brands_ocr_results(vlogger, s3_client, video_id: str, brand_results: List[Dict]):
    @vlogger.log_performance
    async def _save_brands_ocr_results():
        vlogger.logger.info(f"Saving brands OCR results for video: {video_id}")
        key = f'{video_id}/ocr/brands_ocr.json'

        try:
            brand_data = json.dumps(brand_results, indent=2)
            data_size = len(brand_data)

            vlogger.logger.debug(f"Attempting to save brands OCR results to S3 for video: {video_id}")
            await vlogger.log_performance(asyncio.to_thread)(
                s3_client.put_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=key,
                Body=brand_data,
                ContentType='application/json'
            )
            
            vlogger.log_s3_operation("upload", data_size)
            vlogger.logger.info(f"Successfully saved brands OCR results for video: {video_id}. Size: {data_size} bytes")

        except ClientError as e:
            vlogger.logger.error(f"Error saving brands OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            vlogger.logger.error(f"Unexpected error saving brands OCR results for video {video_id}: {str(e)}", exc_info=True)
            raise

    return await _save_brands_ocr_results()
########################################################

## Create and save brands OCR results table for video
########################################################
async def create_and_save_brand_table(vlogger, s3_client, video_id: str, brand_appearances: Dict[str, Set[int]], fps: float):
    @vlogger.log_performance
    async def _create_and_save_brand_table():
        vlogger.logger.info(f"Creating and saving brand table for video: {video_id}")
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
                vlogger.logger.debug(f"Brand '{brand}' added to stats with {len(frame_list)} frames")
            else:
                vlogger.logger.info(f"Discarded brand '{brand}' as it appeared for less than 1 second ({len(frame_list)} frames)")

        vlogger.logger.info(f"Brand table created with {len(brand_stats)} entries for video: {video_id}")

        try:
            brand_table_data = json.dumps(brand_stats, indent=2)
            data_size = len(brand_table_data)
            vlogger.logger.debug(f"Attempting to save brand table to S3 for video: {video_id}")

            # Use asyncio.to_thread for the S3 operation
            await asyncio.to_thread(
                s3_client.put_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/brands_table.json',
                Body=brand_table_data,
                ContentType='application/json'
            )

            vlogger.log_s3_operation("upload", data_size)
            vlogger.logger.info(f"Successfully saved brand table for video: {video_id}. Size: {data_size} bytes")
            return brand_stats
        except ClientError as e:
            vlogger.logger.error(f"Error saving brand table for video {video_id}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            vlogger.logger.error(f"Unexpected error saving brand table for video {video_id}: {str(e)}", exc_info=True)
            raise

    return await _create_and_save_brand_table()
########################################################

## Get video's wordcloud
########################################################
async def get_word_cloud(video_id: str):
    # app_logger.log_info(f"Received request for word cloud of video: {video_id}")
    wordcloud_key = f'{video_id}/ocr/wordcloud.jpg'

    try:
        # app_logger.log_info(f"Attempting to retrieve word cloud from S3 for video: {video_id}")
        response = await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=wordcloud_key
        )
        image_data = response['Body'].read()
        image_size = len(image_data)
        # app_logger.log_info(f"Successfully retrieved word cloud for video {video_id}. Size: {image_size} bytes")

        return StreamingResponse(BytesIO(image_data), media_type="image/jpeg")

    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"Word cloud not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Word cloud not found for video {video_id}")

    except Exception as e:
        app_logger.log_error(f"Error retrieving word cloud for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving word cloud")
########################################################

## Get brands OCR table for video
########################################################
async def get_brands_ocr_table(video_id: str):
    # app_logger.log_info(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_table.json'

    try:
        # app_logger.log_info(f"Attempting to retrieve brand OCR table from S3 for video: {video_id}")
        response = await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=ocr_key
        )
        data = response['Body'].read()
        data_size = len(data)

        brands_table = json.loads(data.decode('utf-8'))
        # app_logger.log_info(f"Successfully retrieved brand OCR table for video {video_id}. Size: {data_size} bytes")
        
        return brands_table

    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"Brands OCR table not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Brands OCR table not found for video {video_id}")
    
    except json.JSONDecodeError as e:
        app_logger.log_error(f"Error decoding JSON for brand OCR table of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing brand OCR table")

    except s3_client.exceptions.ClientError as e:
        app_logger.log_error(f"S3 client error retrieving brand OCR table for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR table from storage")

    except Exception as e:
        app_logger.log_error(f"Unexpected error retrieving brand OCR table for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving brand OCR table")
########################################################

## Get video's OCR results
########################################################
async def get_ocr_results(video_id: str):
    # app_logger.log_info(f"Received request for OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/ocr_results.json'

    try:
        # app_logger.log_info(f"Attempting to retrieve OCR results from S3 for video: {video_id}")
        response = await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=ocr_key
        )
        data = response['Body'].read()
        data_size = len(data)

        ocr_results = json.loads(data.decode('utf-8'))
        # app_logger.log_info(f"Successfully retrieved OCR results for video {video_id}. Size: {data_size} bytes")
        
        return ocr_results

    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"OCR results not found for video {video_id}")
    
    except json.JSONDecodeError as e:
        app_logger.log_error(f"Error decoding JSON for OCR results of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing OCR results")

    except s3_client.exceptions.ClientError as e:
        app_logger.log_error(f"S3 client error retrieving OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving OCR results from storage")

    except Exception as e:
        app_logger.log_error(f"Unexpected error retrieving OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving OCR results")
########################################################

## Get video's processed OCR results
########################################################
async def get_processed_ocr_results(video_id: str):
    # app_logger.log_info(f"Received request for processed OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/processed_ocr.json'

    try:
        # app_logger.log_info(f"Attempting to retrieve processed OCR results from S3 for video: {video_id}")
        response = await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=ocr_key
        )
        data = response['Body'].read()
        data_size = len(data)

        ocr_results = json.loads(data.decode('utf-8'))
        # app_logger.log_info(f"Successfully retrieved processed OCR results for video {video_id}. Size: {data_size} bytes")
        
        return ocr_results

    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"Processed OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail=f"Processed OCR results not found for video {video_id}")
    
    except json.JSONDecodeError as e:
        app_logger.log_error(f"Error decoding JSON for processed OCR results of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing OCR results")

    except s3_client.exceptions.ClientError as e:
        app_logger.log_error(f"S3 client error retrieving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processed OCR results from storage")

    except Exception as e:
        app_logger.log_error(f"Unexpected error retrieving processed OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving processed OCR results")
########################################################

## Get video processed OCR results
########################################################
async def get_brands_ocr_results(video_id: str):
    # app_logger.log_info(f"Received request for brand OCR results of video: {video_id}")
    ocr_key = f'{video_id}/ocr/brands_ocr.json'
    
    try:
        # app_logger.log_info(f"Attempting to retrieve brand OCR results from S3 for video: {video_id}")
        response = await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=ocr_key
        )
        data = response['Body'].read()
        data_size = len(data)

        ocr_results = json.loads(data.decode('utf-8'))
        # app_logger.log_info(f"Successfully retrieved brand OCR results for video {video_id}. Size: {data_size} bytes")
        
        return ocr_results

    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"Brand OCR results not found for video {video_id}")
        raise HTTPException(status_code=404, detail="Brand OCR results not found")
    
    except json.JSONDecodeError as e:
        app_logger.log_error(f"Error decoding JSON for brand OCR results of video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing brand OCR results")

    except s3_client.exceptions.ClientError as e:
        app_logger.log_error(f"S3 client error retrieving brand OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving brand OCR results from storage")

    except Exception as e:
        app_logger.log_error(f"Unexpected error retrieving brand OCR results for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving brand OCR results")
########################################################