import aioboto3
import aiofiles
import asyncio
from botocore.config import Config
from contextlib import asynccontextmanager
from core.config import settings
from botocore.exceptions import ClientError

# Global S3 client
_s3_client = None
_s3_client_lock = asyncio.Lock()

async def initialize_s3_client(use_acceleration=False):
    global _s3_client
    if _s3_client is None:
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
        _s3_client = await session.client('s3', config=s3_config).__aenter__()

async def cleanup_s3_client():
    global _s3_client
    if _s3_client is not None:
        await _s3_client.__aexit__(None, None, None)
        _s3_client = None

@asynccontextmanager
async def get_s3_client(use_acceleration=False):
    """
    Asynchronous context manager to provide an aioboto3 S3 client.
    Ensures proper initialization and reuse of the client.
    """
    global _s3_client
    async with _s3_client_lock:
        if _s3_client is None:
            await initialize_s3_client(use_acceleration)
    try:
        yield _s3_client
    finally:
        # We don't clean up here to allow reuse, cleanup should be called explicitly when shutting down
        pass

async def multipart_upload(file_path, bucket, key, file_size):
    """
    Performs a multipart upload to S3 using the provided file details.
    """
    chunk_size = 1024 * 1024 * settings.MULTIPART_CHUNK_SIZE  # Convert to bytes

    async def upload_part(part_number, start_byte, end_byte, upload_id):
        async with aiofiles.open(file_path, 'rb') as f:
            await f.seek(start_byte)
            part_data = await f.read(end_byte - start_byte)

        try:
            async with get_s3_client() as s3_client:
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

    upload_id = None
    try:
        # Initiate multipart upload
        async with get_s3_client() as s3_client:
            mpu = await s3_client.create_multipart_upload(Bucket=bucket, Key=key)
            upload_id = mpu['UploadId']

            # Prepare parts
            tasks = [
                upload_part(i + 1, start, min(start + chunk_size, file_size), upload_id)
                for i, start in enumerate(range(0, file_size, chunk_size))
            ]

            # Upload parts concurrently
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
        if upload_id:
            async with get_s3_client() as s3_client:
                await s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise

# Make sure to call this function when shutting down your application
async def shutdown():
    await cleanup_s3_client()