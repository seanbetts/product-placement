import os
import cv2
import tempfile
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from contextlib import asynccontextmanager

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
        self._temp_file_path: Optional[str] = None

    @asynccontextmanager
    async def _video_file(self):
        """Context manager for handling the temporary video file."""
        try:
            if self._temp_file_path is None:
                self._temp_file_path = await self._download_from_s3()
            yield self._temp_file_path
        finally:
            await self._cleanup()

    async def _download_from_s3(self) -> str:
        """Download the video from S3 to a temporary location."""
        s3_key = f"{self.details['video_id']}/original.mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
        
        try:
            async with get_s3_client() as s3_client:
                await s3_client.download_file(settings.PROCESSING_BUCKET, s3_key, temp_path)
                return temp_path
        except Exception as e:
            os.unlink(temp_path)
            logger.error(f"Failed to download video from S3: {str(e)}")
            raise ValueError(f"Failed to download video from S3: {str(e)}")

    async def _cleanup(self):
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

    def set_detail(self, key: str, value: Any):
        """Set a specific video detail."""
        if key not in self.details:
            raise ValueError(f"Invalid detail key: {key}")
        self.details[key] = value

    async def _calculate_missing_detail(self, key: str):
        """Calculate a missing detail."""
        try:
            async with self._video_file():
                if key == "file_size":
                    self.details[key] = os.path.getsize(self._temp_file_path)
                else:
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
                    cap.release()
        except Exception as e:
            logger.error(f"Error calculating {key}: {str(e)}")
            raise ValueError(f"Error calculating {key}: {str(e)}")

    @classmethod
    async def create(cls, video_id: str):
        """Create a new VideoDetails instance."""
        return cls(video_id)