from fastapi import APIRouter
from core.logging import AppLogger

# Create a global instance of AppLogger
app_logger = AppLogger()

router = APIRouter()

## HEALTH CHECK ENDPOINT (GET)
## Returns a 200 status if the server is healthy
########################################################
@router.get("/health")
async def health_check():
    app_logger.log_info("Health check called")
    return {"status": "product placement backend ok"}
########################################################