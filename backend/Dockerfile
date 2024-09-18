# syntax=docker/dockerfile:1.4
FROM --platform=linux/arm64 python:3.11-slim as builder

WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libenchant-2-2 \
    fonts-liberation \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Final stage
FROM --platform=linux/arm64 python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libenchant-2-2 \
    fonts-liberation \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup --home /home/appuser appuser
USER appuser

EXPOSE 8080

# Add more logging and error checking
CMD ["sh", "-c", "\
    echo 'Starting container' && \
    echo 'Python version:' && python --version && \
    echo 'Pip version:' && pip --version && \
    echo 'Installed packages:' && pip list && \
    echo 'Current directory:' && pwd && \
    echo 'Directory contents:' && ls -la && \
    if command -v gunicorn &> /dev/null; then \
        echo 'Gunicorn found, starting server' && \
        gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 0 main:app -k uvicorn.workers.UvicornWorker --log-level debug --access-logfile - --error-logfile -; \
    else \
        echo 'Gunicorn not found in PATH' && \
        echo 'PATH:' && echo $PATH && \
        exit 1; \
    fi \
"]