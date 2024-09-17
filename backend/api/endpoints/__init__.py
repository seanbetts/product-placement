from fastapi import APIRouter
from . import audio, download, frames, management, ocr, status, videos

router = APIRouter()

router.include_router(audio.router)
router.include_router(download.router)
router.include_router(frames.router)
router.include_router(management.router)
router.include_router(ocr.router)
router.include_router(status.router)
router.include_router(videos.router)
