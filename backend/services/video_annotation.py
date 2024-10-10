import cv2
import numpy as np
import json
import os
import tempfile
import subprocess
import asyncio
from asyncio import Semaphore
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
from core.config import settings
from core.aws import get_s3_client
from core.s3_upload import save_data_to_s3
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from models.detection_classes import BrandInstance
from core.logging import logger
from utils.utils import find_font

TEMP_DIR = settings.TEMP_DIR
SMOOTHING_WINDOW = settings.SMOOTHING_WINDOW

## Process video frames for annotation
########################################################
async def annotate_video(video_id: str, status_tracker: StatusTracker, video_details: VideoDetails):
    max_concurrent_tasks = settings.MAX_WORKERS
    
    try:
        logger.info(f"Video Processing - Video Annotation - Step 5.2: Loading detection results for video {video_id}")
        async with get_s3_client() as s3_client:
            ocr_results_obj = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/brands_ocr.json'
            )
        ocr_results_data = await ocr_results_obj['Body'].read()
        ocr_results = json.loads(ocr_results_data.decode('utf-8'))
        logger.debug(f"Loaded OCR results for {len(ocr_results)} frames")

        with tempfile.TemporaryDirectory() as temp_dir:
            processed_frames_dir = os.path.join(temp_dir, "processed_frames")
            os.makedirs(processed_frames_dir, exist_ok=True)

            logger.info(f"Video Processing - Video Annotation - Step 5.3: Started annotating of frames for video {video_id}")
            
            total_frames = len(ocr_results)
            processed_frames = 0
            failed_frames = 0

            sem = Semaphore(max_concurrent_tasks)

            async def process_frame_with_semaphore(frame_data):
                async with sem:
                    try:
                        return await process_single_frame(video_id, frame_data, processed_frames_dir)
                    except Exception as e:
                        logger.error(f"Video Processing - Video Annotation: Error in process_frame_with_semaphore: {str(e)}", exc_info=True)
                        return False

            tasks = [process_frame_with_semaphore(frame_data) for frame_data in ocr_results]

            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    result = await task
                    if result:
                        processed_frames += 1
                    else:
                        failed_frames += 1
                except Exception as e:
                    logger.error(f"Video Processing - Video Annotation: Error processing frame {i}/{total_frames} for video {video_id}: {str(e)}", exc_info=True)
                    failed_frames += 1

                if i % 100 == 0 or i == total_frames:
                    logger.info(f"Video Processing - Video Annotation - Step 5.3: Processed {i}/{total_frames} frames ({((i / total_frames) * 100):.0f}%) for video {video_id}")
                    logger.debug(f"Processed {i}/{total_frames} frames for video {video_id}. "
                                        f"Successful: {processed_frames}, Failed: {failed_frames}")
                    progress = (i / total_frames) * 100
                    await status_tracker.update_process_status("annotation", "in_progress", progress)

            logger.info(f"Video Processing - Video Annotation - Step 5.4: Finished annotating of frames for video {video_id}")

            logger.debug(f"Completed frame processing for video {video_id}. "
                                f"Successfully processed: {processed_frames}, Failed: {failed_frames}")

            if processed_frames == 0:
                raise RuntimeError(f"Video Processing - Video Annotation: No frames were successfully processed for video {video_id}")

            frame_files = sorted(os.listdir(processed_frames_dir))
            logger.debug(f"Found {len(frame_files)} processed frames in {processed_frames_dir}")

            if not frame_files:
                raise RuntimeError(f"Video Processing - Video Annotation: No processed frames found in {processed_frames_dir}")

            logger.info(f"Video Processing - Video Annotation - Step 5.5: Started video reconstruction for video {video_id}")
            await reconstruct_video(video_id, temp_dir, video_details)

        await status_tracker.update_process_status("annotation", "complete", 100)

    except Exception as e:
        logger.error(f"Video Processing - Video Annotation: Error in annotate_video for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.set_error(f"Video Processing - Video Annotation: Error in video annotation: {str(e)}")
        raise
########################################################

## Process each individual frame
########################################################
async def process_single_frame(video_id: str, frame_data: dict, processed_frames_dir: str):
    frame_number = None
    try:
        frame_number = int(frame_data['frame_number'])
        logger.debug(f"Video Processing - Video Annotation: Processing frame {frame_number} for video {video_id}")

        original_frame_key = f"{video_id}/frames/{frame_number:06d}.jpg"
        logger.debug(f"Video Processing - Video Annotation: Downloading frame {frame_number} for video {video_id}")

        async with get_s3_client() as s3_client:
            frame_obj = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=original_frame_key
            )
            frame_bytes = await frame_obj['Body'].read()

        logger.debug(f"Video Processing - Video Annotation: Decoding frame {frame_number} for video {video_id}")
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError(f"Video Processing - Video Annotation: Failed to decode frame {frame_number}")

        logger.debug(f"Video Processing - Video Annotation: Successfully downloaded and decoded frame {frame_number}")

        detected_brands = frame_data.get('detected_brands', [])

        logger.debug(f"Video Processing - Video Annotation: Annotating frame {frame_number} for video {video_id}")
        frame = await annotate_frame(
            frame,
            detected_brands,
            show_confidence=settings.SHOW_CONFIDENCE,
            text_bg_opacity=settings.TEXT_BG_OPACITY
        )
        logger.debug(f"Video Processing - Video Annotation: Successfully annotated frame {frame_number}")

        processed_frame_path = os.path.join(processed_frames_dir, f"processed_frame_{frame_number:06d}.jpg")
        logger.debug(f"Video Processing - Video Annotation: Saving annotated frame {frame_number} to {processed_frame_path}")
        cv2.imwrite(processed_frame_path, frame)

        if not os.path.exists(processed_frame_path):
            raise IOError(f"Video Processing - Video Annotation: Failed to save annotated frame {frame_number}")

        logger.debug(f"Video Processing - Video Annotation: Successfully saved processed frame {frame_number} to {processed_frame_path}")

        logger.debug(f"Video Processing - Video Annotation: Uploading annotated frame {frame_number} for video {video_id}")
        async with get_s3_client() as s3_client:
            with open(processed_frame_path, 'rb') as f:
                await s3_client.put_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=f"{video_id}/processed_frames/processed_frame_{frame_number:06d}.jpg",
                    Body=f
                )

        logger.debug(f"Video Processing - Video Annotation: Successfully uploaded annotated frame {frame_number}")
        logger.debug(f"Video Processing - Video Annotation: Completed annotating frame {frame_number} for video {video_id}")
        return True

    except Exception as e:
        logger.error(f"Video Processing - Video Annotation: Error annotating frame {frame_number} for video {video_id}: {str(e)}", exc_info=True)
        return False
########################################################

## Annotate individual video frames
########################################################
async def annotate_frame(
    frame: np.ndarray,
    detected_brands: List[Dict],
    show_confidence: bool = True,
    text_bg_opacity: float = 0.7,
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        logger.error("Video Processing - Video Annotation: Frame must be a numpy array")
        raise ValueError("Video Processing - Video Annotation: Frame must be a numpy array")
    
    logger.debug(f"Video Processing - Video Annotation: Input frame shape: {frame.shape}, dtype: {frame.dtype}")

    # Find the first available preferred font
    font_size = 16
    font_path = await find_font(settings.PREFERRED_FONTS)

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            logger.debug(f"Video Processing - Video Annotation: Loaded font from {font_path}")
        except IOError:
            logger.warning(f"Video Processing - Video Annotation: Failed to load font from {font_path}, using default")
            font = ImageFont.load_default()
    
    if len(detected_brands) > 0:
        logger.debug(f"Video Processing - Video Annotation: Annotating frame with {len(detected_brands)} detected brands")
        
        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.debug(f"Video Processing - Video Annotation: Converted frame to RGB. Shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
        
        pil_image = Image.fromarray(frame_rgb)
        logger.debug(f"Video Processing - Video Annotation: Created PIL Image. Size: {pil_image.size}, mode: {pil_image.mode}")
        
        draw = ImageDraw.Draw(pil_image)
        
        for brand in detected_brands:
            try:
                if 'bounding_box' not in brand or 'vertices' not in brand['bounding_box']:
                    logger.warning(f"Video Processing - Video Annotation: Skipping brand without bounding box: {brand.get('brand', 'Unknown')}")
                    continue  # Skip brands without bounding boxes
                
                vertices = brand['bounding_box']['vertices']
                brand_name = brand.get('brand', '').upper()  # Capitalize the brand name
                confidence = brand.get('brand_match_confidence', 0.0)
                
                logger.debug(f"Video Processing - Video Annotation: Processing brand: {brand_name}, confidence: {confidence}")

                pts = np.array([(int(v['x']), int(v['y'])) for v in vertices], np.int32)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                
                try:
                    color = await get_color_by_confidence(confidence)
                    logger.debug(f"Video Processing - Video Annotation: Color returned by get_color_by_confidence: {color}")
                except Exception as color_error:
                    logger.error(f"Video Processing - Video Annotation: Error in get_color_by_confidence: {str(color_error)}")
                    color = (255, 0, 0)  # Default to red if there's an error
                
                # Draw bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
                logger.debug(f"Video Processing - Video Annotation: Drew bounding box at ({x_min}, {y_min}, {x_max}, {y_max})")

                # Prepare label
                label = f"{brand_name}" + (f" ({confidence:.2f})" if show_confidence else "")
                label_size = draw.textbbox((0, 0), label, font=font)
                label_width = label_size[2] - label_size[0]
                label_height = label_size[3] - label_size[1]

                # Position label box
                label_x = x_min
                label_y = y_min - label_height if y_min > label_height else y_max

                # Draw label background
                label_bg = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                label_bg_draw = ImageDraw.Draw(label_bg)
                label_bg_draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height),
                                        fill=(color[0], color[1], color[2], int(255 * text_bg_opacity)))
                pil_image = Image.alpha_composite(pil_image.convert('RGBA'), label_bg)
                draw = ImageDraw.Draw(pil_image)

                # Draw text
                text_x = label_x + (label_width - label_size[2]) // 2
                text_y = label_y + (label_height - label_size[3]) // 2
                draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))

                logger.debug(f"Video Processing - Video Annotation: Annotated brand: {brand_name} at position ({x_min}, {y_min}, {x_max}, {y_max})")
            except Exception as e:
                logger.error(f"Video Processing - Video Annotation : Error annotating brand {brand.get('brand', 'Unknown')}: {str(e)}")

        logger.debug("Video Processing - Video Annotation: Frame annotation completed")

        # Convert back to OpenCV format
        annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        logger.debug(f"Video Processing - Video Annotation: Converted annotated frame back to BGR. Shape: {annotated_frame.shape}, dtype: {annotated_frame.dtype}")

        return annotated_frame
    else:
        logger.debug("Video Processing - Video Annotation: No brands detected, returning original frame")
        return frame
########################################################

## Reconstruct video with annotations
########################################################
async def reconstruct_video(video_id: str, temp_dir: str, video_details: VideoDetails):
    processed_frames_dir = os.path.join(temp_dir, "processed_frames")
    output_video_path = os.path.join(temp_dir, f"{video_id}_output.mp4")
    final_output_path = os.path.join(temp_dir, f"{video_id}_final.mp4")
    audio_path = os.path.join(temp_dir, f"{video_id}_audio.mp3")

    try:
        # Fetch FPS
        try:
            original_fps = await video_details.get_detail("frames_per_second")
        except Exception as e:
            logger.error(f"Video Processing - Video Annotation: Error fetching processing stats for video {video_id}: {str(e)}", exc_info=True)
            original_fps = 30  # Fallback to 30 fps if we can't get the original
            logger.warning(f"Video Processing - Video Annotation: Using fallback frame rate of {original_fps} fps")

        # Check if processed frames exist
        frame_files = sorted(os.listdir(processed_frames_dir))
        if not frame_files:
            raise RuntimeError(f"Video Processing - Video Annotation: No processed frames found in {processed_frames_dir}")

        logger.debug(f"Found {len(frame_files)} processed frames for video {video_id}")

        frame_pattern = os.path.join(processed_frames_dir, 'processed_frame_%06d.jpg')

        # Verify that at least one frame matching the pattern exists
        if not any(os.path.exists(frame_pattern % i) for i in range(1, 6)):
            raise RuntimeError(f"Video Processing - Video Annotation: No frames matching pattern {frame_pattern} found")

        ffmpeg_command = [
            'ffmpeg',
            '-framerate', str(original_fps),
            '-i', frame_pattern,
            '-c:v', settings.VIDEO_CODEC,
            '-preset', settings.VIDEO_PRESET,
            '-profile:v', settings.VIDEO_PROFILE,
            '-b:v', settings.VIDEO_BITRATE,
            '-pix_fmt', settings.VIDEO_PIXEL_FORMAT,
            output_video_path
        ]

        # Run FFmpeg command and capture output
        logger.debug(f"Running FFmpeg command to create video for {video_id}")
        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_command)}")
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_message = stderr.decode() if stderr else "Unknown error"
            logger.info(f"Video Processing - Video Annotation: FFmpeg command failed for {video_id}. Error: {error_message}")
            logger.debug(f"FFmpeg command: {' '.join(ffmpeg_command)}")
            raise RuntimeError(f"Video Processing - Video Annotation: FFmpeg command failed: {error_message}")
        
        logger.debug(f"FFmpeg output: {stdout.decode()}")

        logger.debug(f"Downloading audio file for video {video_id}")
        async with get_s3_client() as s3_client:
            await s3_client.download_file(
                settings.PROCESSING_BUCKET, 
                f"{video_id}/audio.mp3", 
                audio_path
            )

        combine_command = [
            'ffmpeg',
            '-i', output_video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', settings.AUDIO_CODEC,
            '-b:a', settings.AUDIO_BITRATE,
            final_output_path
        ]

        # Run FFmpeg command to combine video and audio
        logger.debug(f"Video Processing - Video Annotation: Running FFmpeg command to combine video and audio for {video_id}")
        result = await asyncio.subprocess.create_subprocess_exec(
            *combine_command, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, combine_command, stderr)
        logger.debug(f"Video Processing - Video Annotation: FFmpeg combine output: {stdout.decode()}")

        logger.info(f"Video Processing - Video Annotation - Step 5.6: Finished reconstructing annotated video {video_id}")
        
        logger.info(f"Video Processing - Video Annotation - Step 5.7: Uploading annotated video {video_id}")
        
        # Wait for the upload task to complete
        try:
            await save_data_to_s3(video_id, 'processed_video.mp4', final_output_path)
        except asyncio.TimeoutError:
            logger.error(f"Video Processing - Video Annotation - Step 5.7: Upload timeout for video {video_id}")
            raise
        except Exception as upload_error:
            logger.error(f"Video Processing - Video Annotation - Step 5.7: Upload failed for video {video_id}: {str(upload_error)}")
            raise

        logger.info(f"Video Processing - Video Annotation - Step 5.8: Successfully uploaded annotated video {video_id}")

    except RuntimeError as e:
        logger.error(f"Video Processing - Video Annotation: FFmpeg error for {video_id}: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Video Processing - Video Annotation: Error in reconstruct_video for {video_id}: {str(e)}", exc_info=True)
        raise
    finally:
        for file_path in [output_video_path, final_output_path, audio_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Video Processing - Video Annotation: Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Video Processing - Video Annotation: Error removing temporary file {file_path}: {str(e)}")
########################################################

## Get colour by confidence
########################################################
async def get_color_by_confidence(confidence: float) -> Tuple[int, int, int]:
    try:
        logger.debug(f"Entered _get_color_by_confidence with confidence: {confidence}")
        
        # Check if confidence is a valid float
        if not isinstance(confidence, float):
            logger.error(f"Video Processing - Video Annotation: Invalid confidence type: {type(confidence)}")
            raise ValueError(f"Video Processing - Video Annotation: Confidence must be a float, got {type(confidence)}")

        # Ensure confidence is between 0 and 1
        original_confidence = confidence
        conf = max(0, min(1, confidence / 100 if confidence > 1 else confidence))
        logger.debug(f"Adjusted confidence from {original_confidence} to {conf}")

        # Define color ranges
        if conf < 0.5:
            # Red (255, 0, 0) to Yellow (255, 255, 0)
            r = 255
            g = int(255 * (conf * 2))
            b = 0
            logger.debug(f"Confidence < 0.5, using red to yellow range. Color: ({r}, {g}, {b})")
        else:
            # Yellow (255, 255, 0) to Green (0, 255, 0)
            r = int(255 * ((1 - conf) * 2))
            g = 255
            b = 0
            logger.debug(f"Confidence >= 0.5, using yellow to green range. Color: ({r}, {g}, {b})")
        
        return (r, g, b)
    
    except Exception as e:
        logger.error(f"Video Processing - Video Annotation: Error in get_color_by_confidence: {str(e)}", exc_info=True)
        raise
########################################################