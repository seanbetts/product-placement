from fastapi import FastAPI
from api.routes import router
from core.config import settings
from core.logging import logger
from core.middleware import add_middleware
import uvicorn
from core import aws

# Set up FastAPI
app = FastAPI(title=settings.PROJECT_NAME)

# Add middleware
add_middleware(app)

# Include router
app.include_router(router)

# Event handler for application startup
@app.on_event("startup")
async def on_startup():
    logger.info("FastAPI application startup initiated")
    try:
        await aws.initialize_s3_client()
        logger.info("S3 client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing S3 client: {str(e)}")
        raise

# Event handler for application shutdown
@app.on_event("shutdown")
async def on_shutdown():
    logger.info("FastAPI application shutdown initiated")
    try:
        await aws.shutdown()
        logger.info("S3 client shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down S3 client: {str(e)}")

# Run the application
if __name__ == "__main__":
    logger.info(f"Starting server on port {settings.PORT}")
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, log_level="info", workers=settings.MAX_API_WORKERS)