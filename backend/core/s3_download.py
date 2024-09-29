import asyncio
from typing import Optional
from botocore.exceptions import ClientError
from core.aws import get_s3_client
from core.logging import logger
from core.config import settings

async def get_s3_object(key: str) -> Optional[bytes]:
    bucket = settings.PROCESSING_BUCKET
    timeout = 10
    try:
        logger.debug(f"Fetching {key} from S3")
        async with get_s3_client() as s3_client:
            # Check if the object exists
            try:
                await s3_client.head_object(Bucket=bucket, Key=key)
                logger.debug(f"{key} exists in S3")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.error(f"Object not found in S3: bucket={bucket}, key={key}")
                    return None
                else:
                    raise

            # Fetch the object with a timeout
            try:
                logger.debug(f"Downloading {key} from S3")
                obj = await asyncio.wait_for(
                    s3_client.get_object(Bucket=bucket, Key=key),
                    timeout=timeout
                )
                content = await obj['Body'].read()
                logger.debug(f"Successfully downloaded {key} from S3")
                return content
            except asyncio.TimeoutError:
                logger.error(f"Timeout while fetching S3 object: bucket={bucket}, key={key}")
                return None
    except Exception as e:
        logger.error(f"Error fetching S3 object: bucket={bucket}, key={key}, error={str(e)}", exc_info=True)
        raise