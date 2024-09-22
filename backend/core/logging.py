import logging
import time
import psutil
import os
import tempfile
import asyncio
import boto3
import atexit
from botocore.exceptions import ClientError
from functools import wraps
from contextlib import contextmanager
from core.config import settings
from utils.decorators import retry

s3_client = boto3.client('s3')

class StatusCheckFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, 'skip_logging', False)

def setup_root_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addFilter(StatusCheckFilter())

setup_root_logger()

# Global tracker for active loggers
active_loggers = {}

# VideoLogger class for monitoring performance of video uploads and processing
class VideoLogger:
    root_logger = None

    @classmethod
    def setup_root_logger(cls):
        if cls.root_logger is None:
            cls.root_logger = logging.getLogger("VideoProcessor")
            cls.root_logger.setLevel(logging.INFO)

    def __init__(self, video_id, is_api_log=False):
        self.video_id = video_id
        self.is_api_log = is_api_log
        self.log_file = os.path.join(tempfile.gettempdir(), f"{video_id}_performance.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = self._get_logger()
        self.s3_operations = {
            "upload": 0,
            "download": 0,
            "total_upload_size": 0,
            "total_download_size": 0
        }
        self.last_upload_time = time.time()

    def _get_logger(self):
        VideoLogger.setup_root_logger()
        logger = logging.getLogger(f"VideoProcessor.{self.video_id}")
        logger.setLevel(logging.INFO)

        # Add file handler for this specific video ID
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    
    def enable_console_logging(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_performance(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            start_cpu = psutil.cpu_percent()

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                end_cpu = psutil.cpu_percent()

                duration = end_time - start_time
                memory_used = end_memory - start_memory
                cpu_used = end_cpu - start_cpu

                self.logger.info(f"Function {func.__name__} performance:"
                                f"\n\tTime: {duration:.2f} seconds"
                                f"\n\tMemory: {memory_used / (1024 * 1024):.2f} MB"
                                f"\n\tCPU: {cpu_used:.2f}%")
                
                self._check_and_upload_log()

        return wrapper

    def log_s3_operation(self, operation_type, size):
        self.s3_operations[operation_type] += 1
        self.s3_operations[f"total_{operation_type}_size"] += size
        self.logger.info(f"S3 {operation_type}: Count: {self.s3_operations[operation_type]}, "
                         f"Total Size: {self.s3_operations[f'total_{operation_type}_size'] / (1024 * 1024):.2f} MB")
        
        self._check_and_upload_log()

    def _check_and_upload_log(self):
        current_time = time.time()
        if current_time - self.last_upload_time > 300:  # Upload every 5 minutes
            self._upload_log_to_s3()
            self.last_upload_time = current_time

    @retry(exceptions=(ClientError,), tries=3, delay=1, backoff=2)
    def _upload_log_to_s3(self):
        try:
            if not os.path.exists(self.log_file):
                self.logger.warning(f"Log file does not exist: {self.log_file}")
                return
            
            if self.is_api_log:
                s3_key = f"_logs/{os.path.basename(self.log_file)}"
            else:
                s3_key = f"{self.video_id}/{os.path.basename(self.log_file)}"

            s3_client.upload_file(
                self.log_file, 
                settings.PROCESSING_BUCKET, 
                s3_key
            )
            self.logger.info(f"Performance log uploaded to S3")
        except Exception as e:
            self.logger.error(f"Error uploading performance log to S3: {str(e)}")

    def finalize(self):
        self.logger.info(f"Finalizing logging")
        self.logger.info(f"Final S3 operation counts: {self.s3_operations}")
        try:
            self._upload_log_to_s3()
        except Exception as e:
            self.logger.error(f"Failed to upload log to S3: {str(e)}")
        finally:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
                self.logger.info(f"Local log file removed")

@contextmanager
def video_logger(logger_name, is_api_log=False):
    if logger_name in active_loggers:
        logger = active_loggers[logger_name]
    else:
        logger = VideoLogger(logger_name, is_api_log)
        active_loggers[logger_name] = logger
    try:
        yield logger
    finally:
        logger.finalize()
        active_loggers.pop(logger_name, None)

def finalize_all_loggers():
    for video_id, logger in list(active_loggers.items()):
        try:
            logger.finalize()
        except Exception as e:
            print(f"Error finalizing logger for video {video_id}: {str(e)}")
        active_loggers.pop(video_id, None)

# Register the finalization function to run at exit
atexit.register(finalize_all_loggers)

# Add this new function for dual logging
def dual_log(vlogger, app_logger, level, message, **kwargs):
    log_func = getattr(vlogger.logger, level)
    log_func(message, **kwargs)
    
    app_log_func = getattr(app_logger, f"log_{level}")
    app_log_func(message, **kwargs)

# AppLogger class for general application logging
class AppLogger:
    def __init__(self):
        self.log_file = "/app/temp/app-log.log"
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logger = logging.getLogger("app-log")

        if not logger.hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename=self.log_file,
                filemode='w'
            )

            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console.setFormatter(formatter)
            logger.addHandler(console)

        return logger

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message, **kwargs):
        self.logger.error(message, **kwargs)

# Export both loggers
__all__ = ['video_logger', 'AppLogger', 'dual_log']