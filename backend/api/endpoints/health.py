from fastapi import APIRouter
from core.logging import logger

router = APIRouter()

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
########################################################
@router.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "product placement backend ok"}
########################################################