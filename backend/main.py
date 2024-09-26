import os
from fastapi import FastAPI
from api.routes import router
from core.config import settings
from core.logging import AppLogger
from core.middleware import add_middleware
import uvicorn
from core import aws

# Create a global instance of AppLogger
app_logger = AppLogger()

# Set up FastAPI
app = FastAPI(title=settings.PROJECT_NAME)

# Add middleware
add_middleware(app)

# Include router
app.include_router(router)

# Event handler for application startup
@app.on_event("startup")
async def on_startup():
    app_logger.log_info("FastAPI application startup initiated")
    try:
        await aws.initialize_s3_client()
        app_logger.log_info("S3 client initialized successfully")
    except Exception as e:
        app_logger.log_error(f"Error initializing S3 client: {str(e)}")
        # You might want to raise an exception here to prevent the app from starting
        # if the S3 client can't be initialized
        raise

# Event handler for application shutdown
@app.on_event("shutdown")
async def on_shutdown():
    app_logger.log_info("FastAPI application shutdown initiated")
    try:
        await aws.shutdown()
        app_logger.log_info("S3 client shut down successfully")
    except Exception as e:
        app_logger.log_error(f"Error shutting down S3 client: {str(e)}")

# Run the application
if __name__ == "__main__":
    app_logger.log_info(f"Starting server on port {settings.PORT}")
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, log_level="info")