import logging
from uvicorn.logging import AccessFormatter

class CustomAccessFormatter(AccessFormatter):
    def formatMessage(self, record):
        message = super().formatMessage(record)
        if getattr(record, 'skip_logging', False):
            return None  # Skip logging this message
        return message

class SkipAccessFilter(logging.Filter):
    def filter(self, record):
        try:
            headers = record.scope['headers']
            for name, value in headers:
                if name == b'x-skip-logging' and value == b'true':
                    record.skip_logging = True
                    return False
        except Exception:
            pass
        return True