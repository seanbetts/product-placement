import logging
import time
import psutil
import os
import tempfile
import asyncio
import atexit
import inspect
import functools
from contextlib import asynccontextmanager
from core.config import settings
from core.aws import get_s3_client

# Global tracker for active loggers
active_loggers = {}

def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# VideoLogger class for monitoring performance of video uploads and processing
class VideoLogger:
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
        self.upload_queue = asyncio.Queue()
        self.upload_task = None

    def _get_logger(self):
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
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            start_cpu = psutil.cpu_percent()

            try:
                result = await func(*args, **kwargs)
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
                
                await self._queue_log_upload()

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            start_cpu = psutil.cpu_percent()

            try:
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
                
                asyncio.create_task(self._queue_log_upload())

        # Return the appropriate wrapper based on whether the function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
        
    async def _queue_log_upload(self):
        await self.upload_queue.put(time.time())
        if self.upload_task is None or self.upload_task.done():
            self.upload_task = asyncio.create_task(self._process_upload_queue())

    async def _process_upload_queue(self):
        while True:
            current_time = await self.upload_queue.get()
            if current_time - self.last_upload_time > 300:  # Upload every 5 minutes
                await self._upload_log_to_s3()
                self.last_upload_time = current_time
            self.upload_queue.task_done()
            if self.upload_queue.empty():
                break

    async def log_s3_operation(self, operation_type, size):
        self.s3_operations[operation_type] += 1
        self.s3_operations[f"total_{operation_type}_size"] += size
        self.logger.info(f"S3 {operation_type}: Count: {self.s3_operations[operation_type]}, "
                         f"Total Size: {self.s3_operations[f'total_{operation_type}_size'] / (1024 * 1024):.2f} MB")
        
        await self._check_and_upload_log()

    async def _check_and_upload_log(self):
        current_time = time.time()
        if current_time - self.last_upload_time > 300:  # Upload every 5 minutes
            await self._upload_log_to_s3()
            self.last_upload_time = current_time

    async def _upload_log_to_s3(self):
        if not os.path.exists(self.log_file):
            self.logger.warning(f"Log file does not exist: {self.log_file}")
            return
        
        if self.is_api_log:
            s3_key = f"_logs/{os.path.basename(self.log_file)}"
        else:
            s3_key = f"{self.video_id}/{os.path.basename(self.log_file)}"

        try:
            async with get_s3_client() as s3_client:
                await s3_client.upload_file(
                    Filename=self.log_file, 
                    Bucket=settings.PROCESSING_BUCKET, 
                    Key=s3_key
                )
            self.logger.info(f"Performance log uploaded to S3 at {s3_key}")
        except Exception as e:
            self.logger.error(f"Error uploading performance log to S3: {str(e)}")

    async def finalize(self):
        self.logger.info(f"Finalizing logging")
        self.logger.info(f"Final S3 operation counts: {self.s3_operations}")
        try:
            await self._upload_log_to_s3()
        except Exception as e:
            self.logger.error(f"Failed to upload log to S3: {str(e)}")
        finally:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
                self.logger.info(f"Local log file removed")

@asynccontextmanager
async def video_logger(logger_name, is_api_log=False):
    if logger_name in active_loggers:
        logger = active_loggers[logger_name]
    else:
        logger = VideoLogger(logger_name, is_api_log)
        active_loggers[logger_name] = logger
    try:
        yield logger
    finally:
        await logger.finalize()
        active_loggers.pop(logger_name, None)

async def finalize_all_loggers():
    for video_id, logger in list(active_loggers.items()):
        try:
            await logger.finalize()
        except Exception as e:
            print(f"Error finalizing logger for video {video_id}: {str(e)}")
        active_loggers.pop(video_id, None)

def sync_finalize_all_loggers():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Schedule the coroutine to run soon
        asyncio.create_task(finalize_all_loggers())
    else:
        loop.run_until_complete(finalize_all_loggers())

@atexit.register
def at_exit_finalize():
    try:
        run_async(finalize_all_loggers())
    except Exception as e:
        print(f"Error during log finalization at exit: {str(e)}")

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
        logger.setLevel(logging.INFO)

        # Prevent adding multiple handlers if already added
        if not logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(self.log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message, **kwargs):
        self.logger.error(message, **kwargs)

# Export both loggers
__all__ = ['video_logger', 'AppLogger', 'dual_log']