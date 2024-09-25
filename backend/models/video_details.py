# video_details.py

import json
from typing import Dict, Any
import asyncio
from core.aws import get_s3_client
from core.config import settings

class VideoDetails:
    def __init__(self, video_id: str):
        self.details: Dict[str, Any] = {
            "video_id": video_id,
            "video_resolution": None,
            "frames_per_second": None,
            "number_of_frames": None,
            "video_length": None,
            "file_size": None
        }

    def set_detail(self, key: str, value: Any):
        """Set a specific video detail."""
        if key in self.details:
            self.details[key] = value
        else:
            raise ValueError(f"Invalid detail key: {key}")

    def get_detail(self, key: str) -> Any:
        """Get a specific video detail."""
        if key in self.details:
            return self.details[key]
        else:
            raise ValueError(f"Invalid detail key: {key}")

    async def load_from_s3(self):
        """Load video details from S3."""
        try:
            s3_client = await get_s3_client()
            details_key = f'{self.details["video_id"]}/video_details.json'
            response = await s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=details_key)
            self.details.update(json.loads(await response['Body'].read()))
        except Exception as e:
            print(f"Error loading video details from S3: {str(e)}")

    async def save_to_s3(self):
        """Save video details to S3."""
        try:
            s3_client = await get_s3_client()
            details_key = f'{self.details["video_id"]}/video_details.json'
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=details_key,
                Body=json.dumps(self.details),
                ContentType='application/json'
            )
        except Exception as e:
            print(f"Error saving video details to S3: {str(e)}")

    @classmethod
    async def create(cls, video_id: str):
        """Create a new VideoDetails instance and load details from S3."""
        instance = cls(video_id)
        await instance.load_from_s3()
        return instance