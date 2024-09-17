FROM python:3.11

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

COPY ./data ./data

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libenchant-2-2 \
    fonts-liberation \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system appgroup && adduser --system --ingroup appgroup --home /home/appuser appuser

COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8080

# Local command
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-"]

# Server command
CMD ["sh", "-c", "gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 0 main:app -k uvicorn.workers.UvicornWorker --log-level debug --access-logfile - --error-logfile -"]
