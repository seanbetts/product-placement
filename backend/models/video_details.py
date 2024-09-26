import os
import cv2
import json
import tempfile
from typing import Dict, Any
from core.config import settings
from core.aws import get_s3_client

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
        self._temp_file_path = None

    async def _ensure_video_downloaded(self):
        """Ensure the video is downloaded locally before processing."""
        if self._temp_file_path is None:
            self._temp_file_path = await self._download_from_s3()

    async def _download_from_s3(self) -> str:
        """Download the video from S3 to a temporary location."""
        s3_client = await get_s3_client()
        s3_key = f"{self.details['video_id']}/original.mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
        
        try:
            await s3_client.download_file(settings.PROCESSING_BUCKET, s3_key, temp_path)
            return temp_path
        except Exception as e:
            os.unlink(temp_path)
            raise ValueError(f"Failed to download video from S3: {str(e)}")

    def _cleanup(self):
        """Remove the temporary video file."""
        if self._temp_file_path and os.path.exists(self._temp_file_path):
            os.unlink(self._temp_file_path)
            self._temp_file_path = None

    async def get_detail(self, key: str) -> Any:
        """Get a specific video detail, calculating it if it's None."""
        if key not in self.details:
            raise ValueError(f"Invalid detail key: {key}")
        
        if self.details[key] is None:
            await self._calculate_missing_detail(key)
        
        return self.details[key]

    async def _calculate_missing_detail(self, key: str):
        """Calculate a missing detail."""
        await self._ensure_video_downloaded()
        try:
            cap = cv2.VideoCapture(self._temp_file_path)
            if key == "video_resolution":
                self.details[key] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            elif key == "frames_per_second":
                self.details[key] = cap.get(cv2.CAP_PROP_FPS)
            elif key == "number_of_frames":
                self.details[key] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            elif key == "video_length":
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.details[key] = frame_count / fps if fps else 0
            elif key == "file_size":
                self.details[key] = os.path.getsize(self._temp_file_path)
            cap.release()
        finally:
            self._cleanup()

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