from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

class FilteredLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, excluded_paths: list[str]):
        super().__init__(app)
        self.excluded_paths = excluded_paths

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        response = await call_next(request)
        for excluded_path in self.excluded_paths:
            if path.endswith(excluded_path):
                response.headers['X-Skip-Logging'] = 'true'
                break
        return response

def add_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add FilteredLoggingMiddleware
    app.add_middleware(FilteredLoggingMiddleware, excluded_paths=["/video/status"])