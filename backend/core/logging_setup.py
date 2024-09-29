import logging
import os
import watchtower
import boto3
from logging.handlers import RotatingFileHandler
from core.config import settings
from botocore.exceptions import ClientError

def get_log_level(level_str: str) -> int:
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def setup_logging(log_file, log_group="/ecs/product-placement-fastapi-backend-task", stream_name="fastapi-logs"):
    # Get log level from settings
    log_level = get_log_level(settings.LOG_LEVEL)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [L] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S +0000')

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)

    # Create file handler and set level to debug
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    # Add file handler to logger
    logger.addHandler(fh)

    # CloudWatch handler
    try:
        # Create a boto3 client for CloudWatch Logs
        logs_client = boto3.client('logs')
        
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group=log_group,
            stream_name=stream_name,
            use_queues=False,  # This can help with immediate logging
            create_log_group=False,  # Assuming the log group already exists
            log_group_retention_days=30,  # Adjust as needed
            boto3_client=logs_client
        )
        cw_handler.setLevel(log_level)
        cw_handler.setFormatter(formatter)
        logger.addHandler(cw_handler)
        logger.info(f"CloudWatch logging enabled for group: {log_group}")
    except Exception as e:
        logger.error(f"Failed to set up CloudWatch logging: {str(e)}", exc_info=True)

    return logger