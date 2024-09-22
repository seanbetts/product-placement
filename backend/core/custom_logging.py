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
            # Check for 'X-Skip-Logging' header
            headers = record.scope['headers']
            for name, value in headers:
                if name == b'x-skip-logging' and value == b'true':
                    return False
            
            # Check for 'HTTP/1.1' in the log message
            if 'HTTP/1.1' in record.args[1]:
                return False
        except Exception:
            pass
        return True