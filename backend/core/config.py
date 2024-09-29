import json
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

    ## Frame Processing Settings
    DOWNLOAD_CHUNK_SIZE: int = 8388608                      # 8MB chunks
    FRAME_INTERVAL: int = 1                                 # Frame intervals to process, 1 = all
    BATCH_SIZE: int = 30                                    # Number of images per batch
    MAX_WORKERS: int = 30                                   # Max workers for ThreadPoolExecutor
    MAX_CONCURRENT_BATCHES: int = 10                        # Max number of concurrent batches to process
    MAX_CONCURRENT_UPLOADS: int = 10                        # Max number of concurrent uploads
    MAX_QUEUE_SIZE: int = 100                               # Max number of batches in the queue
    STATUS_UPDATE_INTERVAL: int = 3                         # in seconds

    ## OCR Settings
    BRAND_DATABASE_FILE: Path = Field(default="data/brand_database.json")
    BRAND_DATABASE: Dict[str, Dict] = {}                    # Brand database object
    MIN_BRAND_TIME: int = 1                                 # Minimum number of seconds a brand needs to appear
    MAX_CLEANING_CONFIDENCE: int = 67                       # Maximum confidence score required to skip text cleaning
    MIN_WORD_MATCH: int = 80                                # Minimum confidence for applying word corrections
    HIGH_CONFIDENCE_THRESHOLD: int = 80                     # Minimum score for high-confidence detections
    LOW_CONFIDENCE_THRESHOLD: int = 70                      # Minimum score for low-confidence detections
    MIN_BRAND_LENGTH: int = 3                               # Minimum length of a brand name
    MIN_DETECTIONS: int = 2                                 # Minimum number of detections for a brand to be considered
    FRAME_WINDOW: int = 1                                   # Window (in seconds) for checking brand consistency
    TEXT_DIFF_THRESHOLD: int = 60                           # Minimum fuzzy match score between original and matched text
    MIN_TEXT_WIDTH: int = 5                                 # Minimum text width as percentage of video width
    MIN_TEXT_HEIGHT: int = 5                                # Minimum text height as percentage of video height
    INTERPOLATION_CONFIDENCE: int = 70                      # Confidence score for interpolated brand appearances
    INTERPOLATION_LIMIT: int = 15                           # Maximum consecutive interpolated frames allowed
    WORDCLOUD_MINIMUM_CONFIDENCE: int = 70                  # Minium confidence threshold for words to be included in the wordcloud
    MAX_BOUNDING_BOX_MERGE_DISTANCE_PERCENT: float = 0.02   # 2% of frame dimension
    MIN_OVERLAP_RATIO_FOR_MERGE: float = 0.1                # 10% overlap required for automatic merging

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