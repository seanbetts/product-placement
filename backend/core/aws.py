import boto3
from botocore.config import Config
from core.config import settings

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        s3_config = Config(
            retries={'max_attempts': settings.S3_RETRIES, 'mode': 'adaptive'},
            max_pool_connections=settings.MAX_POOL_CONNECTIONS
        )
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION,
            config=s3_config
        )
    return _s3_client