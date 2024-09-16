from typing import Dict
from threading import Lock

# Dictionary to keep track of active uploads
_active_uploads: Dict[str, bool] = {}
_active_uploads_lock = Lock()

def set_upload_active(video_id: str, active: bool = True):
    with _active_uploads_lock:
        _active_uploads[video_id] = active

def is_upload_active(video_id: str) -> bool:
    with _active_uploads_lock:
        return _active_uploads.get(video_id, False)

def remove_upload(video_id: str):
    with _active_uploads_lock:
        _active_uploads.pop(video_id, None)

def get_all_active_uploads() -> Dict[str, bool]:
    with _active_uploads_lock:
        return _active_uploads.copy()