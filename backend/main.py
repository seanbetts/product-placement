import os
from fastapi import FastAPI
from api.routes import router
from core.config import settings
from core.logging import logger
from core.middleware import add_middleware

logger.info("Starting FastAPI server...")

print(f"Main.py - Current working directory: {os.getcwd()}")

# Set up FastAPI
app = FastAPI(title=settings.PROJECT_NAME)

# Add middleware
add_middleware(app)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {settings.PORT}")
    # logger.info(f"Environment variables: {os.environ}")
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, log_level="debug")