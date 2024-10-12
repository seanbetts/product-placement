import json
import psutil
from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class Settings(BaseSettings):
    ## Project Settings
    PROJECT_NAME: str = "Product Placement Backend"         # App name
    PORT: int = 8080                                        # Port for FastAPI endpoint
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]  # add frontend URL
    API_KEY: str                                            # API key for calling FastAPI endpoints
    TEMP_DIR: str = Field(default="/app/temp")              # Temp directory for local file saving
    LOG_LEVEL: str = "INFO"                                 # Log level
    MAX_API_WORKERS: int = 10                               # Max number of workers for the FastAPI app

    # Preferred fonts in order of priority for text rendering
    PREFERRED_FONTS: tuple[str, ...] = (
        'Liberation Sans', 
        'DejaVu Sans', 
        'Roboto', 
        'Open Sans', 
        'Lato', 
        'Noto Sans',
        'Arial', 
        'Helvetica'
    )

    ## AWS env Settings
    AWS_ACCESS_KEY_ID: str                                  # AWS access key ID
    AWS_SECRET_ACCESS_KEY: str                              # AWS seecret access key
    AWS_DEFAULT_REGION: str                                 # AWS region
    PROCESSING_BUCKET: str                                  # s3 bucket
    REKOGNITION_SNS_TOPIC_ARN: str                          # AWS Rekognition SNS topic ARN
    REKOGNITION_ROLE_ARN: str                               # AWS Rekognition role ARN
    SQS_QUEUE_URL: str

    ## Other AWS Settings
    S3_RETRIES: int = 10                                    # Number of retries for s3
    MAX_POOL_CONNECTIONS: int = 50                          # Max pool connections for s3
    MULTIPART_THRESHOLD: int = 25                           # MB file size before mutlipart uploads take over
    MULTIPART_MAX_CONCURRENCY: int = 10                     # MAximum concurrency
    MULTIPART_CHUNK_SIZE: int = 5                           # MB per chunk

    ## Fixed Frame Processing Settings
    DOWNLOAD_CHUNK_SIZE: int = 8388608                      # 8MB chunks
    FRAME_INTERVAL: int = 1                                 # Frame intervals to process, 1 = all
    MAX_CONCURRENT_UPLOADS: int = 10                        # Max number of concurrent uploads
    MAX_QUEUE_SIZE: int = 100                               # Max number of batches in the queue
    STATUS_UPDATE_INTERVAL: int = 3                         # in seconds

    ## Dynamic Frame Processing Settings
    BATCH_SIZE: int = Field(default_factory=lambda: Settings.calculate_optimal_parameters()[0])                 # Number of images per batch
    MAX_WORKERS: int = Field(default_factory=lambda: Settings.calculate_optimal_parameters()[1])                # Max workers for ThreadPoolExecutor
    MAX_CONCURRENT_BATCHES: int = Field(default_factory=lambda: Settings.calculate_optimal_parameters()[2])     # Max number of concurrent batches to process

    ## Brand Detection Settings
    BRAND_DATABASE_FILE: Path = Field(default="data/brand_database.json")
    BRAND_DATABASE: Dict[str, Dict] = {}                    # Brand database object
    OCR_TYPE: str = 'LINE'                                  # Choose 'LINE' or 'WORD' type
    MINIMUM_OCR_CONFIDENCE: int = 30                        # Minimum acceptable confidence score for including a LINE from raw_ocr.json
    MAX_CLEANING_CONFIDENCE: int = 67                       # Maximum confidence score required to skip text cleaning
    MIN_DETECTIONS: int = 2                                 # Minimum number of detections for a brand to be considered
    MIN_BRAND_TIME: float = 0.9                             # Minimum number of seconds a brand needs to appear
    FRAME_WINDOW: int = 1                                   # Window (in seconds) for checking brand consistency
    INTERPOLATION_LIMIT: int = 15                           # Maximum consecutive interpolated frames allowed
    WORDCLOUD_MINIMUM_CONFIDENCE: int = 70                  # Minium confidence threshold for words to be included in the wordcloud
    MAX_BOUNDING_BOX_MERGE_DISTANCE_PERCENT: float = 0.02   # 2% of frame dimension
    
    ## Fuzzy Brand Matching Settings
    MINIMUM_FUZZY_BRAND_MATCH_SCORE: int = 65               # Minimum fuzzy match score between original and matched text
    MIN_TEXT_LENGTH: int = 3                                # Minimum length of text to be considered for brand matching
    MIN_BRAND_LENGTH: int = 3                               # Minimum length of brand name to be considered for matching
    FULL_MATCH_WEIGHT: float = 0.4                          # Weight given to full text match in overall score calculation
    PARTIAL_MATCH_WEIGHT: float = 0.4                       # Weight given to partial text match in overall score calculation
    WORD_MATCH_WEIGHT: float = 0.3                          # Weight given to word-by-word match in overall score calculation
    MIN_FULL_MATCH_SCORE: int = 70                          # Minimum score required for full text match to be considered
    MIN_PARTIAL_MATCH_SCORE: int = 80                       # Minimum score required for partial text match to be considered
    MIN_WORD_MATCH_SCORE: int = 80                          # Minimum score required for word-by-word match to be considered
    LENGTH_PENALTY_FACTOR: float = 0.1                      # Factor to penalize short matches, reducing false positives
    EXACT_MATCH_BONUS: int = 20                             # Bonus for matching a brand name or variation exactly
    CONTAINS_BRAND_BONUS: int = 50                          # Bonus for containing the brand name
    WORD_IN_BRAND_BONUS: int = 30                           # Bonus for the word being part of the whole brand name
    MIN_VERTICAL_OVERLAP_RATIO_FOR_MERGE: float = 0.015     # 1.5% vertical overlap required for merging
    MIN_HORIZONTAL_OVERLAP_RATIO_FOR_MERGE: float = 0.85    # 85% horizontal overlap required for merging
    COMMON_WORDS: set = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}  # Common words to ignore in matching process
    
    # Video post-processing settings
    SMOOTHING_WINDOW: int = 5                               # Smoothing window for
    SHOW_CONFIDENCE: bool = False                           # Show confidence score for detected brands in annotated video
    TEXT_BG_OPACITY: float = 0.7                            # Background opacity for text in annotated video

    # FFmpeg settings
    FFMPEG_TIMEOUT: int = 300                               # Seconds for timeout
    VIDEO_CODEC: str = 'libx264'                            # Annotated video codec used
    VIDEO_PRESET: str = 'medium'                            # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    VIDEO_PROFILE: str = 'high'                             # Options: baseline, main, high
    VIDEO_BITRATE: str = '5M'                               # 5 Mbps
    VIDEO_PIXEL_FORMAT: str = 'yuv420p'                     # Annotated video pixel format
    AUDIO_CODEC: str = 'aac'                                # Annotated video audio codec
    AUDIO_BITRATE: str = '192k'                             # 192 kbps

    @staticmethod
    def calculate_optimal_parameters():
        cpu_count = psutil.cpu_count(logical=False)
        total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        
        # Multiplier for the number of CPU cores to determine the maximum number of workers
        # Higher values allow for more concurrent tasks but may increase CPU contention
        worker_factor = 2.0

        # Fraction of available memory to be used for batches
        # Higher values allocate more memory for batches, potentially increasing throughput but leaving less for other operations
        batch_size_factor = 0.4

        # Minimum number of items in a batch
        # Ensures that batches are not too small, which could lead to excessive overhead
        min_batch_size = 50

        # Maximum number of items in a batch
        # Prevents batches from becoming too large, which could cause memory issues or reduce responsiveness
        max_batch_size = 1000

        # Estimated memory usage per worker in GB
        # Used to calculate the maximum number of workers based on available memory
        # Lower values allow for more workers but may underestimate actual memory usage
        estimated_memory_per_worker = 0.2

        # Minimum number of workers to use, regardless of other calculations
        # Ensures a base level of parallelism
        min_workers = 16

        # Maximum number of workers allowed, regardless of available resources
        # Prevents excessive parallelism which could lead to diminishing returns or system instability
        max_workers_limit = 32

        # Calculate max workers
        max_workers = max(min_workers, min(
            max_workers_limit,
            int(cpu_count * worker_factor),
            int(available_memory / estimated_memory_per_worker)
        ))

        # Calculate batch size
        memory_for_batches = available_memory * batch_size_factor
        batch_size = max(min_batch_size, min(
            max_batch_size,
            int(memory_for_batches / max_workers)
        ))

        # Ensure batch size is at least 100 if we have enough memory
        if memory_for_batches >= 100 * max_workers:
            batch_size = max(batch_size, 100)

        # Calculate max concurrent batches
        max_concurrent_batches = max(4, min(max_workers // 2, int(available_memory / (batch_size * estimated_memory_per_worker))))

        return batch_size, max_workers, max_concurrent_batches

    @field_validator("BRAND_DATABASE_FILE", mode="before")
    @classmethod
    def validate_brand_database_file(cls, v):
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        return v

    def load_brand_database(self):        
        try:
            with self.BRAND_DATABASE_FILE.open('r') as f:
                self.BRAND_DATABASE = json.load(f)
        except FileNotFoundError:
            print(f"Brand database file not found: {self.BRAND_DATABASE_FILE}")
            self.BRAND_DATABASE = {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {self.BRAND_DATABASE_FILE}")
            self.BRAND_DATABASE = {}
        except Exception as e:
            print(f"Unexpected error loading brand database: {str(e)}")
            self.BRAND_DATABASE = {}

    def model_post_init(self, __context: Any) -> None:
        self.load_brand_database()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

settings = Settings()