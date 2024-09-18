from fastapi import APIRouter, Depends
from api.endpoints import audio, download, frames, health, management, ocr, status, videos
from core.auth import get_api_key

router = APIRouter()

# Include the health check router without API key dependency
router.include_router(health.router, tags=["health"])

# Apply the API key dependency to all routes
router.include_router(audio.router, tags=["audio"], dependencies=[Depends(get_api_key)])
router.include_router(download.router, tags=["download"], dependencies=[Depends(get_api_key)])
router.include_router(frames.router, tags=["frames"], dependencies=[Depends(get_api_key)])
router.include_router(management.router, tags=["management"], dependencies=[Depends(get_api_key)])
router.include_router(ocr.router, tags=["ocr"], dependencies=[Depends(get_api_key)])
router.include_router(status.router, tags=["status"], dependencies=[Depends(get_api_key)])
router.include_router(videos.router, tags=["upload"], dependencies=[Depends(get_api_key)])