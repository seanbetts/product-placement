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

# Ensure /tmp has correct permissions
RUN mkdir -p /tmp && chmod 777 /tmp

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
ENV TEMP=/app/temp

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

# Create a dedicated temp directory and set permissions
RUN mkdir -p /app/temp && chmod 777 /app/temp

RUN addgroup --system appgroup && adduser --system --ingroup appgroup --home /home/appuser appuser
RUN chown -R appuser:appgroup /app/temp

USER appuser

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-"]