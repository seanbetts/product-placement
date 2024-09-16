import os
import json
import datetime
import time
import tempfile
import asyncio
import subprocess
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from services import audio_processing, frames_processing, status_processing
from services.ocr_processing import main_ocr_processing
from utils import utils
import boto3
from botocore.exceptions import ClientError

## Processes an uploaded video
########################################################
async def run_video_processing(video_id: str):
    logger.info(f"Starting to process video: {video_id}")
    video_key = f'{video_id}/original.mp4'

    s3_client = get_s3_client()

    status_tracker = StatusTracker(video_id)
    status_tracker.update_s3_status(s3_client)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        s3_client.download_file(settings.PROCESSING_BUCKET, video_key, temp_video.name)
        temp_video_path = temp_video.name

    try:
        total_start_time = time.time()
        
        # Start status update task
        status_update_task = asyncio.create_task(status_processing.periodic_status_update(video_id, status_tracker, s3_client))

        # Run video frame processing and audio extraction in parallel
        video_task = asyncio.create_task(frames_processing.process_video_frames(temp_video_path, video_id, s3_client, status_tracker))
        audio_task = asyncio.create_task(extract_audio_from_video(temp_video_path, video_id, s3_client, status_tracker))

        # Wait for video and audio tasks to complete
        video_stats, audio_stats = await asyncio.gather(video_task, audio_task)

        if status_tracker.status.get("error"):
            logger.error(f"Error encountered during video/audio processing: {status_tracker.status['error']}")
            return

        # Start OCR processing after video frames are available
        ocr_task = asyncio.create_task(main_ocr_processing.process_ocr(video_id, status_tracker, s3_client))

        # Start transcription after audio extraction is complete
        transcription_stats = await audio_processing.transcribe_audio(video_id, s3_client, float(video_stats['video_length'].split()[0]), status_tracker)
        
        if status_tracker.status.get("error"):
            logger.error(f"Error encountered during transcription: {status_tracker.status['error']}")
            return

        # Wait for OCR processing to complete
        ocr_stats = await ocr_task

        if status_tracker.status.get("error"):
            logger.error(f"Error encountered during OCR processing: {status_tracker.status['error']}")
            return

        # Post-process OCR results
        video_resolution = await utils.get_video_resolution(video_id)
        brand_results = await main_ocr_processing.post_process_ocr(video_id, video_stats['video_fps'], video_resolution, s3_client)

        # Wait for all processes to complete
        await status_tracker.wait_for_completion()

        # Cancel the status update task
        status_update_task.cancel()

        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        processing_stats = {
            "video_id": video_id,
            "video_length": video_stats['video_length'],
            "video": video_stats,
            "audio": audio_stats,
            "transcription": transcription_stats,
            "ocr": ocr_stats,
            "total_processing_start_time": datetime.datetime.fromtimestamp(total_start_time).isoformat(),
            "total_processing_end_time": datetime.datetime.fromtimestamp(total_end_time).isoformat(),
            "total_processing_time": f"{total_processing_time:.2f} seconds",
            "total_processing_speed": f"{(float(video_stats['video_length'].split()[0]) / total_processing_time * 100):.1f}% of real-time"
        }

        # Save processing stats to a new file
        stats_key = f'{video_id}/processing_stats.json'
        s3_client.put_object(Bucket=settings.PROCESSING_BUCKET, Key=stats_key, 
                             Body=json.dumps(processing_stats, indent=2), 
                             ContentType='application/json')

        # Update final status
        status_tracker.status["status"] = "complete"
        status_tracker.update_s3_status(s3_client)

        # Mark video as completed
        status_processing.mark_video_as_completed(video_id)

        # logger.info(f"Completed processing video: {video_id}")
        # logger.info(f"Total processing time: {total_processing_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        status_tracker.set_error(str(e))
        status_tracker.update_s3_status(s3_client)
    finally:
        os.unlink(temp_video_path)
########################################################

## Extracts audio from video
########################################################
async def extract_audio_from_video(video_path: str, video_id: str, s3_client, status_tracker: StatusTracker):
    logger.info(f"Extracting audio for video: {video_id}")
    audio_path = f"/tmp/{video_id}_audio.mp3"
    start_time = time.time()
    status_tracker.update_process_status("audio_extraction", "in_progress", 0)

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

        # Upload audio file to S3
        with open(audio_path, 'rb') as audio_file:
            s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/audio.mp3',
                Body=audio_file,
                ContentType='audio/mpeg'
            )
        # logger.info(f"Audio extracted and uploaded for video: {video_id}")

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

        status_tracker.update_process_status("audio_extraction", "complete", 100)

        return {
            "audio_length": f"{audio_duration:.2f} seconds",
            "audio_processing_time": processing_time,
            "audio_processing_speed": processing_speed
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting audio for video {video_id}: {str(e)}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        status_tracker.update_process_status("audio_extraction", "error", 0)
    except Exception as e:
        logger.error(f"Unexpected error extracting audio for video {video_id}: {str(e)}", exc_info=True)
        status_tracker.update_process_status("audio_extraction", "error", 0)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {
        "audio_length": "Unknown",
        "audio_processing_time": time.time() - start_time,
        "audio_processing_speed": 0
    }
########################################################

## Deletes a video
########################################################
async def delete_video(video_id: str):
    s3_client = get_s3_client()
    
    try:
        bucket = settings.PROCESSING_BUCKET
        prefix = f'{video_id}/'
        
        s3_resource = boto3.resource('s3')
        bucket_resource = s3_resource.Bucket(bucket)
        
        # Delete objects and capture any errors
        delete_errors = []
        for obj in bucket_resource.objects.filter(Prefix=prefix):
            try:
                obj.delete()
            except Exception as e:
                delete_errors.append(f"Error deleting {obj.key}: {str(e)}")

        if delete_errors:
            logger.error(f"Partial deletion for video {video_id}. Errors: {', '.join(delete_errors)}")
            raise Exception("Partial deletion occurred")

        logger.info(f"Successfully deleted all objects for video {video_id}")

        await remove_from_completed_videos(video_id)

    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        raise
########################################################

# Adds a video to _completed_videos.json
########################################################
def update_completed_videos_list(video_id: str):
    completed_videos_key = '_completed_videos.json'
    s3_client = get_s3_client()
    
    try:
        # Try to get the existing list
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=completed_videos_key)
        completed_videos = json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # If the file doesn't exist, start with an empty list
            completed_videos = []
        else:
            raise

    # Add the new video ID if it's not already in the list
    if video_id not in completed_videos:
        completed_videos.append(video_id)

    # Upload the updated list back to S3
    s3_client.put_object(
        Bucket=settings.PROCESSING_BUCKET,
        Key=completed_videos_key,
        Body=json.dumps(completed_videos),
        ContentType='application/json'
    )

    # logger.info(f"Added video {video_id} to completed videos list")
########################################################

# Removes a video from _completed_videos.json
########################################################
async def remove_from_completed_videos(video_id: str):
    s3_client = get_s3_client()
    completed_videos_key = '_completed_videos.json'
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=completed_videos_key)
        completed_videos = json.loads(response['Body'].read().decode('utf-8'))
        
        if video_id in completed_videos:
            completed_videos.remove(video_id)
            
            s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=completed_videos_key,
                Body=json.dumps(completed_videos),
                ContentType='application/json'
            )
            logger.info(f"Removed video {video_id} from completed videos list")
    except Exception as e:
        logger.error(f"Error updating completed videos list for video {video_id}: {str(e)}")
        # Don't raise here, as this is a secondary operation
########################################################