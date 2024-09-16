from fastapi import APIRouter
from api.endpoints import audio, download, frames, management, ocr, status, upload

router = APIRouter()

router.include_router(audio.router, tags=["audio"])
router.include_router(download.router, tags=["download"])
router.include_router(frames.router, tags=["frames"])
router.include_router(management.router, tags=["management"])
router.include_router(ocr.router, tags=["ocr"])
router.include_router(status.router, tags=["status"])
router.include_router(upload.router, tags=["upload"])