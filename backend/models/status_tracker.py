import time
import json
import asyncio
from typing import Dict, Any
from core.aws import get_s3_client
from core.config import settings

class StatusTracker:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.start_time = time.time()
        self.status: Dict[str, Any] = {
            "video_id": video_id,
            "status": "processing",
            "progress": 0,
            "video_processing": {"status": "pending", "progress": 0},
            "audio_extraction": {"status": "pending", "progress": 0},
            "transcription": {"status": "pending", "progress": 0},
            "ocr": {"status": "pending", "progress": 0}
        }
        self.process_events = {
            "video_processing": asyncio.Event(),
            "audio_extraction": asyncio.Event(),
            "transcription": asyncio.Event(),
            "ocr": asyncio.Event()
        }
        self.status["error"] = None

    def update_process_status(self, process: str, status: str, progress: float):
        self.status[process]["status"] = status
        self.status[process]["progress"] = progress
        if status == "complete":
            self.process_events[process].set()

    def calculate_overall_progress(self):
        total_progress = sum(self.status[process]["progress"] for process in self.process_events)
        self.status["progress"] = total_progress / len(self.process_events)

    async def wait_for_completion(self):
        await asyncio.gather(*[event.wait() for event in self.process_events.values()])
        self.status["status"] = "complete"

    def get_status(self):
        elapsed_time = time.time() - self.start_time
        self.status["elapsed_time"] = f"{elapsed_time:.2f} seconds"
        return self.status

    def update_s3_status(self, s3_client):
        current_status = self.get_status()
        status_key = f'{self.video_id}/status.json'
        s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=status_key,
            Body=json.dumps(current_status),
            ContentType='application/json'
        )

    def set_error(self, error_message: str):
        s3_client = get_s3_client()
        self.status["error"] = error_message
        self.status["status"] = "error"
        self.update_s3_status(s3_client)