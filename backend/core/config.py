import os
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class Settings(BaseSettings):
    ## Project Settings
    PROJECT_NAME: str = "Product Placement Backend"
    PORT: int = 8080  # Port for FastAPI endpoint
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]  # add frontend URL
    API_KEY: str
    TEMP_DIR: str = Field(default="/app/temp")

    ## AWS env Settings
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str  # AWS region
    PROCESSING_BUCKET: str  # s3 bucket
    REKOGNITION_SNS_TOPIC_ARN: str
    REKOGNITION_ROLE_ARN: str
    SQS_QUEUE_URL: str

    ## Other AWS Settings
    S3_RETRIES: int = 10
    MAX_POOL_CONNECTIONS: int = 50  # Max pool connections for s3

    ## Processing Settings
    BATCH_SIZE: int = 30
    FRAME_INTERVAL: int = 1
    STATUS_UPDATE_INTERVAL: int = 3  # in seconds
    MAX_WORKERS: int = 10

    ## OCR Settings
    MIN_BRAND_TIME: int = 1  # minimum number of seconds a brand needs to appear
    BRAND_DATABASE_FILE: Path = Field(default="data/brand_database.json")
    BRAND_DATABASE: Dict[str, Dict] = {}
    HIGH_CONFIDENCE_THRESHOLD: int = 90    # Minimum score for high-confidence detections
    LOW_CONFIDENCE_THRESHOLD: int = 80     # Minimum score for low-confidence detections
    MIN_BRAND_LENGTH: int = 3              # Minimum length of a brand name
    MIN_DETECTIONS: int = 2                # Minimum number of detections for a brand to be considered
    FRAME_WINDOW: int = 1                  # Window (in seconds) for checking brand consistency
    TEXT_DIFF_THRESHOLD: int = 60          # Minimum fuzzy match score between original and matched text
    MIN_TEXT_WIDTH: int = 5                # Minimum text width as percentage of video width
    MIN_TEXT_HEIGHT: int = 5               # Minimum text height as percentage of video height
    INTERPOLATION_CONFIDENCE: int = 70     # Confidence score for interpolated brand appearances

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

# Add these lines at the end of config.py
settings = Settings()