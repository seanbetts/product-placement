from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

class FilteredLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, excluded_paths: list[str], excluded_options_paths: list[str]):
        super().__init__(app)
        self.excluded_paths = excluded_paths
        self.excluded_options_paths = excluded_options_paths

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        response = await call_next(request)
        
        should_skip = any([
            any(path.endswith(excluded) for excluded in self.excluded_paths),
            method == "OPTIONS" and any(excluded in path for excluded in self.excluded_options_paths)
        ])
        
        if should_skip:
            response.headers['X-Skip-Logging'] = 'true'
        
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
    app.add_middleware(
        FilteredLoggingMiddleware, 
        excluded_paths=["/video/status"],
        excluded_options_paths=[
            "/images/first-frame",
            "/images/all-frames",
            "/video/processing-stats",
            "/transcript"
        ]
    )