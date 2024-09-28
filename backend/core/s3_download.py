import asyncio
from typing import Optional
from botocore.exceptions import ClientError
from core.aws import get_s3_client
from core.logging import AppLogger, dual_log
from core.config import settings

# Create a global instance of AppLogger
app_logger = AppLogger()

async def get_s3_object(vlogger, key: str) -> Optional[bytes]:
    @vlogger.log_performance
    async def _get_s3_object(k: str) -> Optional[bytes]:
        bucket = settings.PROCESSING_BUCKET
        timeout = 10
        try:
            # vlogger.logger.debug(f"Fetching {k} from S3")
            async with get_s3_client() as s3_client:
                # Check if the object exists
                try:
                    await s3_client.head_object(Bucket=bucket, Key=k)
                    # vlogger.logger.debug(f"{k} exists in S3")
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        dual_log(vlogger, app_logger, 'error', f"Object not found in S3: bucket={bucket}, key={k}")
                        return None
                    else:
                        raise

                # Fetch the object with a timeout
                try:
                    # vlogger.logger.debug(f"Downloading {k} from S3")
                    obj = await asyncio.wait_for(
                        s3_client.get_object(Bucket=bucket, Key=k),
                        timeout=timeout
                    )
                    content = await obj['Body'].read()
                    # vlogger.logger.debug(f"Successfully downloaded {k} from S3")
                    return content
                except asyncio.TimeoutError:
                    dual_log(vlogger, app_logger, 'error', f"Timeout while fetching S3 object: bucket={bucket}, key={k}")
                    return None
        except Exception as e:
            dual_log(vlogger, app_logger, 'error', f"Error fetching S3 object: bucket={bucket}, key={k}, error={str(e)}", exc_info=True)
            raise

    return await _get_s3_object(key)