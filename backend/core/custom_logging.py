import logging
from uvicorn.logging import AccessFormatter
import re

class CustomAccessFormatter(AccessFormatter):
    def formatMessage(self, record):
        if getattr(record, 'skip_logging', False):
            return None
        message = super().formatMessage(record)
        return message

class SkipAccessFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.video_status_pattern = re.compile(r'/[^/]+/video/status$')

    def filter(self, record):
        try:
            path = record.scope['path']
            method = record.scope['method']
            
            # Filter out /video/status GET and OPTIONS requests
            if self.video_status_pattern.search(path) and method in ['GET', 'OPTIONS']:
                return False
            
            # Filter out specific OPTIONS requests
            if method == 'OPTIONS' and any(excluded in path for excluded in [
                '/images/first-frame',
                '/images/all-frames',
                '/video/processing-stats',
                '/transcript'
            ]):
                return False
            
        except Exception:
            pass
        return True