import cv2
import os
import json
import tempfile
import uuid
import logging
import datetime
import time
import asyncio
import subprocess
import ffmpeg
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from google.auth import default
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from dotenv import load_dotenv
from io import BytesIO
from typing import List
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables (this will work locally, but not affect GCP environment)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
PROCESSING_BUCKET = os.getenv('PROCESSING_BUCKET')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '20'))
FRAME_INTERVAL = int(os.getenv('FRAME_INTERVAL', '1'))
STATUS_UPDATE_INTERVAL = int(os.getenv('STATUS_UPDATE_INTERVAL', '10'))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))

# Custom transport with larger connection pool
class CustomTransport(AuthorizedSession):
    def __init__(self, credentials):
        super().__init__(credentials)
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"]
        )
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry_strategy)
        self.mount("https://", adapter)

# Set up Google Cloud Storage client with proper authentication and custom transport
try:
    if GOOGLE_APPLICATION_CREDENTIALS:
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    else:
        # Use default credentials (this will work in GCP)
        credentials, _ = default()
    
    custom_transport = CustomTransport(credentials)
    storage_client = storage.Client(credentials=credentials, _http=custom_transport)
except Exception as e:
    logger.error(f"Error setting up Google Cloud Storage client: {str(e)}")
    raise

@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "ok"}

@app.post("/upload")
async def upload_video(video: UploadFile, background_tasks: BackgroundTasks):
    logger.info(f"Received upload request for file: {video.filename}")
    video_id = str(uuid.uuid4())
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    blob = bucket.blob(f'{video_id}/original.mp4')

    try:
        logger.info(f"Uploading file to bucket: {PROCESSING_BUCKET}")
        blob.upload_from_file(video.file, content_type=video.content_type)
        logger.info(f"File uploaded successfully. Video ID: {video_id}")
        
        # Schedule the video processing task
        background_tasks.add_task(run_video_processing, video_id)
        
        return {"video_id": video_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    logger.info(f"Received status request for video ID: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    status_blob = bucket.blob(f'{video_id}/status.json')
    
    if status_blob.exists():
        status_data = json.loads(status_blob.download_as_string())
        logger.info(f"Status for video ID {video_id}: {status_data}")
        return status_data
    else:
        logger.warning(f"Status not found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Video status not found")

@app.get("/processed-videos")
async def get_processed_videos():
    logger.info("Received request for processed videos")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    processed_videos = []

    # List all blobs in the bucket without a delimiter
    blobs = list(bucket.list_blobs())
    logger.info(f"Found {len(blobs)} blobs in the bucket")

    for blob in blobs:
        # Identify potential video folders by the presence of a status.json file
        if blob.name.endswith('status.json'):
            video_id = blob.name.rsplit('/', 2)[-2]
            logger.info(f"Found status.json for video: {video_id}")
            
            status_blob = bucket.blob(blob.name)
            if status_blob.exists():
                try:
                    status_data = json.loads(status_blob.download_as_string())
                    
                    if status_data.get('status') == 'complete':
                        logger.info(f"Video {video_id} is complete, adding to processed videos")
                        video_data = {
                            'video_id': video_id,
                            'details': status_data.get('details', {})
                        }
                        processed_videos.append(video_data)
                    else:
                        logger.info(f"Video {video_id} is not complete, status: {status_data.get('status')}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse status.json for video {video_id}")
            else:
                logger.info(f"No status file found for video: {video_id}")

    logger.info(f"Returning {len(processed_videos)} processed videos")
    return processed_videos

@app.get("/video-frame/{video_id}")
async def get_video_frame(video_id: str):
    logger.info(f"Received request for first frame of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    
    first_frame_blob = bucket.blob(f'{video_id}/frames/000000.jpg')
    if first_frame_blob.exists():
        frame_data = first_frame_blob.download_as_bytes()
        return StreamingResponse(BytesIO(frame_data), media_type="image/jpeg")
    else:
        raise HTTPException(status_code=404, detail="First frame not found")
    
@app.get("/video/{video_id}")
async def get_video_details(video_id: str):
    logger.info(f"Received request for video details: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    status_blob = bucket.blob(f'{video_id}/status.json')

    if status_blob.exists():
        status_data = json.loads(status_blob.download_as_string())
        return status_data
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/video/{video_id}/frames")
async def get_video_frames(video_id: str) -> List[dict]:
    logger.info(f"Received request for video frames: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    frames_prefix = f'{video_id}/frames/'
    blobs = bucket.list_blobs(prefix=frames_prefix)
    
    frames = []
    for blob in blobs:
        frame_number = int(blob.name.split('/')[-1].split('.')[0])
        if frame_number % 50 == 0:
            frames.append({
                "number": frame_number,
                "url": blob.generate_signed_url(version="v4", expiration=3600, method="GET")
            })
    
    return sorted(frames, key=lambda x: x["number"])

@app.get("/video/{video_id}/transcript")
async def get_transcript(video_id: str):
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    transcript_blob = bucket.blob(f'{video_id}/transcripts/transcript.json')
    
    if transcript_blob.exists():
        transcript = json.loads(transcript_blob.download_as_string())
        return transcript
    else:
        raise HTTPException(status_code=404, detail="Transcript not found")

@app.get("/video/{video_id}/download/{file_type}")
async def download_file(video_id: str, file_type: str):
    logger.info(f"Received download request for {file_type} of video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    
    if file_type == "video":
        blob = bucket.blob(f'{video_id}/original.mp4')
        filename = f"{video_id}_video.mp4"
    elif file_type == "audio":
        blob = bucket.blob(f'{video_id}/audio.mp3')
        filename = f"{video_id}_audio.mp3"
    elif file_type == "transcript":
        blob = bucket.blob(f'{video_id}/transcripts/transcript.txt')
        filename = f"{video_id}_transcript.txt"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if blob.exists():
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        blob.download_to_filename(temp_file.name)
        return FileResponse(temp_file.name, media_type='application/octet-stream', filename=filename)
    else:
        raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} not found")

async def run_video_processing(video_id: str):
    try:
        await process_video(video_id)
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        update_status(video_id, "error", {"error": str(e)})

async def process_video(video_id: str):
    logger.info(f"Starting to process video: {video_id}")
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    video_blob = bucket.blob(f'{video_id}/original.mp4')

    update_status(video_id, "processing", {
        "video_id": video_id,
        "total_processing_start_time": datetime.datetime.now().isoformat()
    })

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        video_blob.download_to_filename(temp_video.name)
        temp_video_path = temp_video.name

    try:
        total_start_time = time.time()

        # Run video frame processing and audio extraction in parallel
        video_task = asyncio.create_task(process_video_frames(temp_video_path, video_id, bucket))
        audio_task = asyncio.create_task(extract_audio(temp_video_path, video_id, bucket))

        # Wait for both tasks to complete
        video_stats, audio_stats = await asyncio.gather(video_task, audio_task)

        # Start transcription only after audio extraction is complete
        transcription_start_time = time.time()
        transcription_stats = await transcribe_audio(video_id, bucket, float(video_stats['video_length'].split()[0]))
        transcription_processing_time = time.time() - transcription_start_time

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        

        stats = {
            "video_id": video_id,
            "video_length": video_stats['video_length'],
            "video": {
                "total_frames": video_stats['total_frames'],
                "extracted_frames": video_stats['extracted_frames'],
                "video_fps": video_stats['video_fps'],
                "video_processing_time": f"{video_stats['video_processing_time']:.2f} seconds",
                "video_processing_fps": f"{video_stats['video_processing_fps']:.2f}",
                "video_processing_speed": f"{video_stats['video_processing_speed']:.1f}% of real-time",
            },
            "audio": {
                "audio_length": audio_stats['audio_length'],
                "audio_processing_time": f"{audio_stats['audio_processing_time']:.2f} seconds",
                "audio_processing_speed": f"{audio_stats['audio_processing_speed']:.1f}% of real-time",
            },
            "transcription": {
                "transcription_processing_time": f"{transcription_stats['transcription_time']:.2f} seconds",
                "word_count": transcription_stats['word_count'],
                "confidence": f"{transcription_stats['overall_confidence']:.1f}%",
                "transcription_speed": f"{transcription_stats['transcription_speed']:.1f}% of real-time"
            },
            "total_processing_start_time": datetime.datetime.fromtimestamp(total_start_time).isoformat(),
            "total_processing_end_time": datetime.datetime.fromtimestamp(total_end_time).isoformat(),
            "total_processing_time": f"{(total_end_time - total_start_time):.2f} seconds",
            "total_processing_speed": f"{(float(video_stats['video_length'].split()[0]) / total_processing_time * 100):.1f}% of real-time"
        }

        update_status(video_id, "complete", stats)

        logger.info(f"Completed processing video: {video_id}")
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        update_status(video_id, "error", {"error": str(e)})
    finally:
        os.unlink(temp_video_path)

async def process_video_frames(video_path: str, video_id: str, bucket: storage.Bucket):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frame_number = 0
    batches = []
    current_batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            if current_batch:
                batches.append(current_batch)
            break

        if frame_number % FRAME_INTERVAL == 0:
            current_batch.append((frame, frame_number))

            if len(current_batch) == BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []

        frame_number += 1

        if frame_number % (STATUS_UPDATE_INTERVAL * int(fps)) == 0:
            progress = (frame_number / frame_count) * 100
            update_status(video_id, "processing", {
                "video": {
                    "progress": f"{progress:.2f}%",
                    "frames_processed": f'{frame_number} frames of {frame_count} total frames',
                }
            })

    cap.release()

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, batch, video_id, bucket) for batch in batches]
        await asyncio.get_event_loop().run_in_executor(None, concurrent.futures.wait, futures)

    processing_time = time.time() - start_time
    extracted_frames = frame_number // FRAME_INTERVAL
    processing_fps = extracted_frames / processing_time
    processing_speed = (processing_fps / fps) * 100

    return {
        "video_length": f'{duration:.1f} seconds',
        "total_frames": frame_count,
        "extracted_frames": extracted_frames,
        "video_fps": round(fps, 2),
        "video_processing_time": processing_time,
        "video_processing_fps": processing_fps,
        "video_processing_speed": processing_speed
    }

async def extract_audio(video_path: str, video_id: str, bucket: storage.Bucket):
    logger.info(f"Extracting audio for video: {video_id}")
    audio_path = f"/tmp/{video_id}_audio.mp3"
    start_time = time.time()
    try:
        command = [
            'ffmpeg',
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            audio_path
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stderr.decode())

        audio_blob = bucket.blob(f'{video_id}/audio.mp3')
        audio_blob.upload_from_filename(audio_path)
        logger.info(f"Audio extracted and uploaded for video: {video_id}")
        
        # Extract audio duration using ffprobe
        duration_command = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            audio_path
        ]
        duration_process = await asyncio.create_subprocess_exec(
            *duration_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await duration_process.communicate()
        audio_duration = float(stdout.decode().strip())
        
        processing_time = time.time() - start_time
        processing_speed = (audio_duration / processing_time) * 100
        
        return {
            "audio_length": f"{audio_duration:.2f} seconds",
            "audio_processing_time": processing_time,
            "audio_processing_speed": processing_speed
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting audio for video {video_id}: {str(e)}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error extracting audio for video {video_id}: {str(e)}", exc_info=True)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    return {
        "audio_length": "Unknown",
        "audio_processing_time": time.time() - start_time,
        "audio_processing_speed": 0
    }

async def transcribe_audio(video_id: str, bucket: storage.Bucket, video_length: float):
    logger.info(f"Transcribing audio for video: {video_id}")
    audio_blob = bucket.blob(f'{video_id}/audio.mp3')

    if not audio_blob.exists():
        logger.warning(f"Audio file not found for video: {video_id}")
        return None

    try:
        # Instantiate the Speech client
        speech_client = SpeechClient()

        # Set up the recognition config
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config={},
            features=cloud_speech.RecognitionFeatures(
                enable_word_confidence=True,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            ),
            model="long",
            language_codes=["en-US"],
        )

        # Set up the audio file metadata
        audio_gcs_uri = f"gs://{PROCESSING_BUCKET}/{video_id}/audio.mp3"
        files = [cloud_speech.BatchRecognizeFileMetadata(uri=audio_gcs_uri)]

        # Set up the output config
        output_config = cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(
                uri=f"gs://{PROCESSING_BUCKET}/{video_id}/transcripts/"
            ),
        )

        # Create the batch recognition request
        request = cloud_speech.BatchRecognizeRequest(
            recognizer=f"projects/{GCP_PROJECT_ID}/locations/global/recognizers/_",
            config=config,
            files=files,
            recognition_output_config=output_config,
        )

        # Start the batch recognition operation
        transcription_start_time = time.time()
        operation = speech_client.batch_recognize(request=request)
        logger.info(f"Started batch recognition for video {video_id}")

        # Wait for the operation to complete with a timeout of 2x video length
        timeout = video_length * 2
        response = await asyncio.to_thread(operation.result, timeout=timeout)
        transcription_end_time = time.time()
        logger.info(f"Batch recognition completed for video {video_id}")

        # Process the response and create transcripts
        plain_transcript, json_transcript, word_count, overall_confidence = await process_transcription_response(bucket, video_id)

        # Upload plain transcript
        plain_transcript_blob = bucket.blob(f'{video_id}/transcripts/transcript.txt')
        plain_transcript_blob.upload_from_string(plain_transcript)

        # Upload JSON transcript
        json_transcript_blob = bucket.blob(f'{video_id}/transcripts/transcript.json')
        json_transcript_blob.upload_from_string(json.dumps(json_transcript, indent=2))

        logger.info(f"Transcripts uploaded for video: {video_id}")

        logger.info(f"Video length: {video_length}")

        # Calculate transcription stats
        transcription_time = transcription_end_time - transcription_start_time
        transcription_speed = (video_length / transcription_time) * 100 if transcription_time > 0 else 0
        overall_confidence = overall_confidence * 100

        return {
            "word_count": word_count,
            "transcription_time": transcription_time,
            "transcription_speed": transcription_speed,
            "overall_confidence": overall_confidence
        }

    except Exception as e:
        logger.error(f"Error transcribing audio for video {video_id}: {str(e)}", exc_info=True)
        return None

async def process_transcription_response(bucket: storage.Bucket, video_id: str):
    plain_transcript = ""
    json_transcript = []
    word_count = 0
    total_confidence = 0

    try:
        # Find the transcript JSON file
        transcript_blobs = list(bucket.list_blobs(prefix=f"{video_id}/transcripts/audio_transcript_"))
        if not transcript_blobs:
            raise FileNotFoundError(f"No transcript file found for video {video_id}")
        
        transcript_blob = transcript_blobs[0]
        transcript_content = transcript_blob.download_as_text()
        transcript_data = json.loads(transcript_content)

        for result in transcript_data.get('results', []):
            for alternative in result.get('alternatives', []):
                transcript = alternative.get('transcript', '')
                plain_transcript += transcript + " "

                for word in alternative.get('words', []):
                    word_count += 1
                    confidence = word.get('confidence', 0)
                    total_confidence += confidence
                    json_transcript.append({
                        "start_time": word.get('startOffset', '0s').rstrip('s'),
                        "end_time": word.get('endOffset', '0s').rstrip('s'),
                        "word": word.get('word', ''),
                        "confidence": confidence
                    })

        # Clean up the plain transcript
        plain_transcript = plain_transcript.strip()

        # Calculate overall confidence score
        overall_confidence = total_confidence / word_count if word_count > 0 else 0

    except Exception as e:
        logger.error(f"Error processing transcription response for video {video_id}: {str(e)}", exc_info=True)
        plain_transcript = "Error processing transcription."
        json_transcript = []
        word_count = 0
        overall_confidence = 0

    return plain_transcript, json_transcript, word_count, overall_confidence

def process_batch(batch, video_id, bucket):
    try:
        uploads = []
        for frame, frame_number in batch:
            frame_filename = f'{frame_number:06d}.jpg'
            _, buffer = cv2.imencode('.jpg', frame)
            frame_blob = bucket.blob(f'{video_id}/frames/{frame_filename}')
            uploads.append((frame_blob, buffer.tobytes()))

        # Perform batch upload
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(uploads), 10)) as executor:
            list(executor.map(lambda x: x[0].upload_from_string(x[1], content_type='image/jpeg'), uploads))
        
    except Exception as e:
        logger.error(f"Error processing batch for video {video_id}: {str(e)}")

def update_status(video_id: str, status: str, details: dict):
    bucket = storage_client.bucket(PROCESSING_BUCKET)
    status_blob = bucket.blob(f'{video_id}/status.json')
    
    # Read existing status if it exists
    if status_blob.exists():
        existing_status = json.loads(status_blob.download_as_string())
    else:
        existing_status = {"details": {}}
    
    # Update status and last_updated
    existing_status["status"] = status
    existing_status["last_updated"] = datetime.datetime.utcnow().isoformat()
    
    # Update details
    existing_status["details"].update(details)
    
    # Upload updated status
    status_blob.upload_from_string(json.dumps(existing_status), content_type='application/json')

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)