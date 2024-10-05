import os
import json
import datetime
import time
import tempfile
import asyncio
import subprocess
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client, multipart_upload
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from services import audio_processing, frames_processing, status_processing, object_detection, video_annotation
from services.ocr_processing import main_ocr_processing
import boto3
from botocore.exceptions import ClientError

## Processes an uploaded video
########################################################
async def run_video_processing(video_id: str):
    try:
        logger.info(f"Video Processing: Starting to process video: {video_id}")
        video_details = await VideoDetails.create(video_id)
        video_key = f'{video_id}/original.mp4'

        status_tracker = StatusTracker(video_id)
        await status_tracker.update_s3_status()

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            async with get_s3_client() as s3_client:
                await s3_client.download_file(settings.PROCESSING_BUCKET, video_key, temp_video.name)
            temp_video_path = temp_video.name

        try:
            total_start_time = time.time()
            
            # Start status update task
            status_update_task = asyncio.create_task(status_processing.periodic_status_update(video_id, status_tracker))

            # Step 1: Run video frame processing and audio extraction in parallel
            logger.info(f"Video Processing - Starting video and audio processing for video: {video_id}")
            await status_tracker.update_process_status("video_processing", "in_progress", 0)
            await status_tracker.update_process_status("audio_extraction", "in_progress", 0)
            video_task = asyncio.create_task(frames_processing.process_video_frames(temp_video_path, video_id, status_tracker, video_details))
            audio_task = asyncio.create_task(extract_audio_from_video(temp_video_path, video_id, status_tracker, video_details))

            # Wait for audio task to complete
            audio_stats = await audio_task

            if status_tracker.status.get("error"):
                logger.error(f"Video Processing - Error encountered during audio processing: {status_tracker.status['error']}")
                await status_tracker.update_s3_status()
                return

            # Step 2: Start transcription immediately after audio extraction
            logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 2.1: Started transcription for video: {video_id}")
            await status_tracker.update_process_status("transcription", "in_progress", 0)
            transcription_task = asyncio.create_task(audio_processing.transcribe_audio(video_id, status_tracker, video_details))

            # Wait for video task to complete
            video_stats = await video_task

            if status_tracker.status.get("error"):
                logger.error(f"Video Processing - Error encountered during video processing: {status_tracker.status['error']}")
                await status_tracker.update_s3_status()
                return

            # Step 3: Start OCR processing after video frames are available
            logger.info(f"Video Processing - Thread 1 - Image Processing - Step 2.1: Started OCR processing for video: {video_id}")
            await status_tracker.update_process_status("ocr", "in_progress", 0)
            ocr_task = asyncio.create_task(main_ocr_processing.process_ocr(video_id, status_tracker, video_details))

            # Wait for transcription and OCR tasks to complete, but handle them separately
            transcription_stats = None
            ocr_stats = None

            pending = {transcription_task, ocr_task}
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task == transcription_task:
                        try:
                            transcription_stats = task.result()
                            await status_tracker.update_process_status("transcription", "complete", 100)
                            logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 2.6: Transcription completed for video: {video_id}")
                            logger.info(f"Video Processing - Thread 2: Audio Processing completed for video: {video_id}")
                        except Exception as e:
                            logger.error(f"Video Processing - Thread 2 - Audio Processing: Error in transcription: {str(e)}")
                            await status_tracker.set_error(f"Video Processing - Thread 2 - Audio Processing: Transcription error: {str(e)}")
                    elif task == ocr_task:
                        try:
                            ocr_stats = task.result()
                            await status_tracker.update_process_status("ocr", "complete", 80)
                            logger.info(f"Video Processing - Thread 1 - Image Processing - Step 2.4: Completed OCR processing for video: {video_id}")
                            logger.info(f"Video Processing - Thread 1: Image Processing completed for video: {video_id}")
                        except Exception as e:
                            logger.error(f"Video Processing - Thread 1 - Image Processing - Step 2.4: Error in OCR processing: {str(e)}")
                            await status_tracker.set_error(f"Video Processing - Thread 1 - Image Processing - Step 2.4: OCR processing error: {str(e)}")

            if status_tracker.status.get("error"):
                logger.error(f"Video Processing - Error encountered during trasnscription and OCR processing: {status_tracker.status['error']}")
                await status_tracker.update_s3_status()
                return

            # Step 4: Brand detection
            logger.info(f"Video Processing - Brand Detection - Step 1.1: Starting brand detection for video: {video_id}")
            brand_results = await main_ocr_processing.brand_detection(video_id, status_tracker, video_details)
            await status_tracker.update_process_status("ocr", "complete", 100)

            # Step 5: Object detection
            logger.info(f"Video Processing - Object Detection - Step 1.1: Starting object detection for video: {video_id}")
            annotation_objects = await object_detection.detect_objects(video_id, brand_results, status_tracker, video_details)
            await status_tracker.update_process_status("objects", "complete", 100)

            # Step 6: Video annotation
            logger.info(f"Video Processing - Video Annotation - Step 1.1: Starting annotation for video: {video_id}")
            await video_annotation.annotate_video(video_id, status_tracker, video_details)
            await status_tracker.update_process_status("annotation", "complete", 100)

            # Wait for all processes to complete
            try:
                await asyncio.wait_for(status_tracker.wait_for_completion(), timeout=3600)  # 1 hour timeout
            except asyncio.TimeoutError:
                await status_tracker.set_error("Video Processing: Processing timed out")
                logger.error(f"Video Processing: Processing timed out for video {video_id}")
                return

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
            stats_json = json.dumps(processing_stats, indent=2)
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=stats_key,
                Body=stats_json,
                ContentType='application/json'
            )

            # Update final status
            await status_tracker.update_process_status("progress", None, 100)
            await status_tracker.update_process_status("status", "complete")

            # Mark video as completed
            await status_processing.mark_video_as_completed(video_id)

            logger.info(f"Video Processing: Completed processing video: {video_id}")
            logger.info(f"Video Processing: Total processing time: {total_processing_time:.2f} seconds")

            # Cancel the status update task
            status_update_task.cancel()

        except Exception as e:
            logger.error(f"Video Processing: Error processing video {video_id}: {str(e)}", exc_info=True)
            await status_tracker.set_error(str(e))
        finally:
            os.unlink(temp_video_path)
    except Exception as e:
        logger.error(f"Error in run_video_processing for video_id {video_id}: {str(e)}")
        logger.error(f"Error in run_video_processing for video_id {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## Extracts audio from video
########################################################
async def extract_audio_from_video(video_path: str, video_id: str, status_tracker: StatusTracker, video_details: VideoDetails):
    logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Extracting audio for video: {video_id}")
    audio_path = f"/tmp/{video_id}_audio.mp3"
    start_time = time.time()
    FFMPEG_TIMEOUT = settings.FFMPEG_TIMEOUT

    try:
        # Estimate total time based on video file size
        video_size = await video_details.get_detail("file_size")
        estimated_total_time = max(1, video_size / 1000000)  # Rough estimate: 1 second per MB, minimum 1 second

        progress_queue = asyncio.Queue()

        async def update_progress():
            while True:
                progress = await progress_queue.get()
                if progress is None:
                    break
                await status_tracker.update_process_status("audio_extraction", "in_progress", progress)
                logger.debug(f"Video Processing: Audio extraction progress for video {video_id}: {progress:.2f}%")

        progress_task = asyncio.create_task(update_progress())

        async def run_ffmpeg():
            command = [
                'ffmpeg',
                '-i', video_path,
                '-q:a', '0',
                '-map', 'a',
                '-progress', 'pipe:1',
                audio_path
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            async def read_output():
                try:
                    while True:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                        if not line:
                            break
                        line = line.decode().strip()
                        if line.startswith("out_time_ms="):
                            time_ms = int(line.split("=")[1])
                            progress = min(100, (time_ms / 1000000) / estimated_total_time * 100)
                            await progress_queue.put(progress)
                except asyncio.TimeoutError:
                    pass  # This is expected when there's no output for a while
                except Exception as e:
                    logger.error(f"Error reading FFmpeg output: {str(e)}")

            output_task = asyncio.create_task(read_output())

            try:
                await asyncio.wait_for(process.wait(), timeout=FFMPEG_TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"FFmpeg process timed out after {FFMPEG_TIMEOUT} seconds")
            finally:
                output_task.cancel()
                try:
                    await output_task
                except asyncio.CancelledError:
                    pass

            if process.returncode != 0:
                stderr = await process.stderr.read()
                raise subprocess.CalledProcessError(process.returncode, command, stderr.decode())

        await run_ffmpeg()

        # Upload audio file to S3 using multipart upload
        s3_key = f'{video_id}/audio.mp3'
        
        try:
            await multipart_upload(audio_path, settings.PROCESSING_BUCKET, s3_key, video_size)
            logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Audio extraction completed for video: {video_id}")
        except Exception as e:
            logger.error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Error uploading audio to S3 for video {video_id}: {str(e)}")
            raise

        # Signal the progress_task to stop
        await progress_queue.put(None)
        await progress_task

        async def get_audio_duration():
            duration_command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            process = await asyncio.create_subprocess_exec(
                *duration_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)  # 30 seconds timeout
                return float(stdout.decode().strip())
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError("ffprobe process timed out after 30 seconds")

        audio_duration = await get_audio_duration()
        processing_time = time.time() - start_time
        processing_speed = (audio_duration / processing_time) * 100

        await status_tracker.update_process_status("audio_extraction", "complete", 100)

        return {
            "audio_length": f"{audio_duration:.2f} seconds",
            "audio_processing_time": processing_time,
            "audio_processing_speed": processing_speed
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: FFmpeg error extracting audio from video {video_id}: {str(e)}")
        logger.error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: FFmpeg stderr: {e.stderr}")
        await status_tracker.set_error(f"FFmpeg error extracting audio from video {video_id}.")
    except TimeoutError as e:
        logger.error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Timeout error processing audio for video {video_id}: {str(e)}")
        await status_tracker.set_error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Timeout error processing audio for video {video_id}.")
    except Exception as e:
        logger.error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Unexpected error extracting audio from video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.set_error(f"Video Processing - Thread 2 - Audio Processing - Step 1.1: Unexpected error extracting audio from video {video_id}.")
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
    logger.info(f"Video Deleteion: Starting deletion process for video: {video_id}")
    
    try:
        bucket = settings.PROCESSING_BUCKET
        prefix = f'{video_id}/'
        s3_resource = boto3.resource('s3')
        bucket_resource = s3_resource.Bucket(bucket)
        
        # Delete objects and capture any errors
        delete_errors = []
        deleted_objects_count = 0
        total_size_deleted = 0

        async def delete_object(obj):
            try:
                size = obj.size
                obj.delete()
                return True, size
            except Exception as e:
                return False, f"Video Deleteion: Error deleting {obj.key}: {str(e)}"

        for obj in bucket_resource.objects.filter(Prefix=prefix):
            success, result = await delete_object(obj)
            if success:
                deleted_objects_count += 1
                total_size_deleted += result
            else:
                delete_errors.append(result)

        if delete_errors:
            logger.error(f"Video Deleteion: Partial deletion for video {video_id}. Errors: {', '.join(delete_errors)}")
            raise Exception("Video Deleteion: Partial deletion occurred")

        logger.info(f"Video Deleteion: Successfully deleted all objects for video {video_id}")
        logger.info(f"Video Deleteion: Deleted {deleted_objects_count} objects, total size: {total_size_deleted / (1024*1024):.2f} MB")

        await remove_from_completed_videos(video_id)

    except Exception as e:
        logger.error(f"Video Deleteion: Error deleting video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

# Adds a video to _completed_videos.json
########################################################
async def update_completed_videos_list(video_id: str):
    completed_videos_key = '_completed_videos.json'
    
    try:
        # Try to get the existing list
        logger.debug(f"Attempting to retrieve existing completed videos list")

        try:
            async with get_s3_client() as s3_client:
                response = await s3_client.get_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=completed_videos_key
                )
            data = await response['Body'].read()
            completed_videos = json.loads(data.decode('utf-8'))
            logger.debug(f"Retrieved existing list with {len(completed_videos)} videos")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # If the file doesn't exist, start with an empty list
                logger.info("No existing completed videos list found. Starting with empty list.")
                completed_videos = []
            else:
                raise

        # Add the new video ID if it's not already in the list
        if video_id not in completed_videos:
            completed_videos.append(video_id)
            logger.debug(f"Added video {video_id} to completed videos list")
        else:
            logger.warning(f"Video {video_id} already in completed videos list")

        # Upload the updated list back to S3
        updated_list_json = json.dumps(completed_videos)
        await s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=completed_videos_key,
            Body=updated_list_json,
            ContentType='application/json'
        )
        logger.debug(f"Updated completed videos list uploaded to S3. Total videos: {len(completed_videos)}")

    except Exception as e:
        logger.error(f"Error updating completed videos list: {str(e)}", exc_info=True)
        raise
########################################################

# Removes a video from _completed_videos.json
########################################################
async def remove_from_completed_videos(video_id: str):
    completed_videos_key = '_completed_videos.json'
    
    try:
        # Try to get the existing list
        logger.debug(f"Attempting to retrieve existing completed videos list")

        try:
            async with get_s3_client() as s3_client:
                response = await s3_client.get_object(
                    Bucket=settings.PROCESSING_BUCKET, 
                    Key=completed_videos_key
                )
            data = await response['Body'].read()
            completed_videos = json.loads(data.decode('utf-8'))
            logger.debug(f"Retrieved existing list with {len(completed_videos)} videos")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error("Completed videos list not found. Nothing to remove from.")
                return
            else:
                raise

        if video_id in completed_videos:
            completed_videos.remove(video_id)
            logger.debug(f"Removed video {video_id} from completed videos list")

            # Upload the updated list back to S3
            updated_list_json = json.dumps(completed_videos)
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=completed_videos_key,
                Body=updated_list_json, 
                ContentType='application/json'
            )
            logger.debug(f"Updated completed videos list uploaded to S3. Total videos: {len(completed_videos)}")
        else:
            logger.warning(f"Video {video_id} not found in completed videos list. No action taken.")

    except Exception as e:
        logger.error(f"Error updating completed videos list for video {video_id}: {str(e)}", exc_info=True)
        # Don't raise here, as this is a secondary operation
########################################################