# Video Processing App

This application processes video files using SAM 2 (Segment Anything Model 2) for object detection and tracking.

## Setup

1. Install dependencies:
pip install -r requirements.txt

2. Set up environment variables:
- PROCESSING_BUCKET: Name of your Google Cloud Storage bucket

3. Run the application:
python main.py

## API Endpoints

- POST /upload: Upload a video file for processing
- GET /status/{video_id}: Check the processing status of a video

For more details, refer to the API documentation.