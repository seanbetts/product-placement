import asyncio
import boto3
import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError
from core.config import settings

s3_client = None
s3_client_lock = asyncio.Lock()
s3_client_sync = None

async def get_s3_client(use_acceleration=False):
    global s3_client
    async with s3_client_lock:
        if s3_client is None:
            s3_config = Config(
                retries={'max_attempts': settings.S3_RETRIES, 'mode': 'adaptive'},
                max_pool_connections=settings.MAX_POOL_CONNECTIONS,
                s3={'use_accelerate_endpoint': use_acceleration}
            )
            session = aioboto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_DEFAULT_REGION,
            )
            # Create the client and enter its async context
            s3_client = await session.client('s3', config=s3_config).__aenter__()
    return s3_client

async def close_s3_client():
    global s3_client
    if s3_client is not None:
        await s3_client.__aexit__(None, None, None)
        s3_client = None

async def multipart_upload(file_path, bucket, key, file_size):
    s3_client = await get_s3_client()
    chunk_size = 1024 * 1024 * settings.MULTIPART_CHUNK_SIZE  # Convert to bytes
    
    async def upload_part(part_number, start_byte, end_byte, upload_id):
        with open(file_path, 'rb') as f:
            f.seek(start_byte)
            part_data = f.read(end_byte - start_byte)
        
        try:
            response = await s3_client.upload_part(
                Bucket=bucket,
                Key=key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=part_data
            )
            return {'PartNumber': part_number, 'ETag': response['ETag']}
        except ClientError as e:
            print(f"Error uploading part {part_number}: {e}")
            raise

    try:
        # Initiate multipart upload
        mpu = await s3_client.create_multipart_upload(Bucket=bucket, Key=key)
        upload_id = mpu['UploadId']

        # Prepare parts
        parts = []
        tasks = []

        for i, start_byte in enumerate(range(0, file_size, chunk_size)):
            part_number = i + 1
            end_byte = min(start_byte + chunk_size, file_size)
            tasks.append(upload_part(part_number, start_byte, end_byte, upload_id))

        # Upload parts
        completed_parts = await asyncio.gather(*tasks)

        # Complete multipart upload
        await s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={'Parts': completed_parts}
        )

    except Exception as e:
        print(f"Error in multipart upload: {e}")
        if 'upload_id' in locals():
            await s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise

def get_s3_client_sync(use_acceleration=False):
    global s3_client_sync
    if s3_client_sync is None:
        s3_config = Config(
            retries={'max_attempts': settings.S3_RETRIES, 'mode': 'adaptive'},
            max_pool_connections=settings.MAX_POOL_CONNECTIONS,
            s3={'use_accelerate_endpoint': use_acceleration}
        )
        s3_client_sync = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION,
            config=s3_config
        )
    return s3_client_sync
