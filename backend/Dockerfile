# Use official Python image as the base image
FROM python:3.9

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install pip and upgrade to the latest version as root
RUN pip install --upgrade pip

# Copy and install Python dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies for OpenCV as root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libenchant-2-2 \
    fonts-liberation \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN addgroup --system appgroup && adduser --system --ingroup appgroup --home /home/appuser appuser
USER appuser

# Copy the rest of the application code
COPY --chown=appuser:appgroup . .

# Specify the command to run the app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app -k uvicorn.workers.UvicornWorker
