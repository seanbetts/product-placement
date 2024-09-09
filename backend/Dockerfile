FROM python:3.11

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

COPY keyfile.json .

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

USER appuser

COPY --chown=appuser:appgroup . .

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app -k uvicorn.workers.UvicornWorker --log-level debug --access-logfile - --error-logfile -