import os
import json
import aiofiles
from typing import Any, Union
from fastapi import HTTPException
from core.config import settings
from core.aws import get_s3_client
from core.logging import logger
from botocore.exceptions import ClientError

## Save data to S3
########################################################
async def save_data_to_s3(video_id: str, filename: str, data: Union[Any, str]):
    logger.debug(f"Saving {filename} to S3 for video: {video_id}")
    FILE_DIRECTORIES = {
        "processed_ocr.json": "ocr",
        "brands_ocr.json": "ocr",
        "raw_object_detection_results.json": "object_detection",
        "original.mp4": None,
        "processed_video.mp4": None
    }

    if filename not in FILE_DIRECTORIES:
        raise HTTPException(status_code=400, detail="Invalid filename")

    directory = FILE_DIRECTORIES[filename]
    key = f'{video_id}/{directory}/{filename}' if directory else f'{video_id}/{filename}'

    try:
        is_json = filename.endswith('.json')
        if is_json:
            processed_data = json.dumps(data, indent=2)
            content_type = 'application/json'
            data_size = len(processed_data)
            upload_func = upload_data
        else:
            if isinstance(data, str) and os.path.isfile(data):  # It's a file path
                content_type = 'application/octet-stream'  # Or use mimetypes.guess_type(filename)[0]
                data_size = os.path.getsize(data)
                upload_func = upload_file
            else:
                processed_data = data
                content_type = 'application/octet-stream'
                data_size = len(data) if isinstance(data, (bytes, bytearray)) else 0
                upload_func = upload_data

        logger.debug(f"Attempting to save {filename} to S3 for video: {video_id}")
        
        async with get_s3_client() as s3_client:
            await upload_func(s3_client, settings.PROCESSING_BUCKET, key, data, content_type)

        logger.debug("Successfully saved {filename} to S3 for video: {video_id}. Size: {data_size} bytes")
    
    except ClientError as e:
        logger.error(f"Error saving {filename} to S3 for video {video_id}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving {filename} to S3 for video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## xxx
########################################################
async def upload_data(s3_client, bucket, key, data, content_type):
    await s3_client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
########################################################

## xxx
########################################################
async def upload_file(s3_client, bucket, key, file_path, content_type):
    try:
        file_size = os.path.getsize(file_path)
        part_size = 8 * 1024 * 1024  # 8 MB part size

        # Initiate multipart upload
        mpu = await s3_client.create_multipart_upload(Bucket=bucket, Key=key, ContentType=content_type)

        parts = []
        async with aiofiles.open(file_path, 'rb') as file:
            part_number = 1
            while True:
                data = await file.read(part_size)
                if not data:
                    break
                
                part = await s3_client.upload_part(Bucket=bucket, Key=key, PartNumber=part_number,
                                                   UploadId=mpu['UploadId'], Body=data)
                parts.append({'PartNumber': part_number, 'ETag': part['ETag']})
                part_number += 1

        # Complete multipart upload
        await s3_client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=mpu['UploadId'],
                                                  MultipartUpload={'Parts': parts})

    except Exception as e:
        # Abort multipart upload if there was an error
        if 'mpu' in locals():
            await s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=mpu['UploadId'])
        raise
########################################################