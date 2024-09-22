import os
import logging
from fastapi import FastAPI
from api.routes import router
from core.config import settings
from core.logging import AppLogger
from core.middleware import add_middleware
from core.custom_logging import CustomAccessFormatter, SkipAccessFilter
import uvicorn

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

# Event handler for application shutdown
@app.on_event("shutdown")
async def on_shutdown():
    app_logger.log_info("FastAPI application shutdown initiated")

# Configure Uvicorn logging
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(SkipAccessFilter())

# Run the application
if __name__ == "__main__":
    app_logger.log_info(f"Starting server on port {settings.PORT}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=settings.PORT, 
        log_level="info",
        log_config={
            "version": 1,
            "formatters": {
                "access": {
                    "()": CustomAccessFormatter,
                    "fmt": '%(client_addr)s - "%(request_line)s" %(status_code)s'
                }
            },
            "handlers": {
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }
    )