import cv2
import numpy as np
import json
import os
import tempfile
import subprocess
import asyncio
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
from core.config import settings
from core.aws import get_s3_client
from utils.decorators import retry
from core.logging import AppLogger, dual_log
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create a global instance of AppLogger
app_logger = AppLogger()


# Create a global instance of s3_client
s3_client = get_s3_client()

TEMP_DIR = settings.TEMP_DIR
SMOOTHING_WINDOW = settings.SMOOTHING_WINDOW

## Process video frames for annotation
########################################################
async def process_video_frames(vlogger, video_id: str):
    @vlogger.log_performance
    async def _process_video_frames():
        dual_log(vlogger, app_logger, 'info', f"Starting post-processing for video {video_id}")
        
        try:
            # Load OCR results
            vlogger.logger.info(f"Loading OCR results for video {video_id}")
            ocr_results_obj = await vlogger.log_performance(asyncio.to_thread)(
                s3_client.get_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/brands_ocr.json'
            )
            ocr_results_data = await ocr_results_obj['Body'].read()
            vlogger.log_s3_operation("download", len(ocr_results_data))
            ocr_results = json.loads(ocr_results_data.decode('utf-8'))
            vlogger.logger.info(f"Loaded OCR results for {len(ocr_results)} frames")

            with tempfile.TemporaryDirectory() as temp_dir:
                processed_frames_dir = os.path.join(temp_dir, "processed_frames")
                os.makedirs(processed_frames_dir, exist_ok=True)
                
                # Process frames in parallel
                vlogger.logger.info(f"Starting parallel processing of frames for video {video_id}")
                with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
                    futures = [executor.submit(process_single_frame, vlogger, video_id, frame_data, s3_client, processed_frames_dir)
                               for frame_data in ocr_results]
                    total_frames = len(futures)
                    processed_frames = 0
                    for i, future in enumerate(as_completed(futures), 1):
                        try:
                            future.result()
                            processed_frames += 1
                            if processed_frames % 100 == 0 or processed_frames == total_frames:
                                vlogger.logger.info(f"Processed {processed_frames}/{total_frames} frames for video {video_id}")
                        except Exception as e:
                            vlogger.logger.error(f"Error processing frame {i}/{total_frames} for video {video_id}: {str(e)}", exc_info=True)

                # Reconstruct video from processed frames
                dual_log(vlogger, app_logger, 'info', f"Starting video reconstruction for video {video_id}")
                await reconstruct_video(vlogger, video_id, temp_dir)

        except Exception as e:
            dual_log(vlogger, app_logger, 'error', f"Error in process_video_frames for video {video_id}: {str(e)}", exc_info=True)
            raise
        finally:
            dual_log(vlogger, app_logger, 'error', f"Completed video post-processing for video {video_id}")

    await _process_video_frames()
########################################################

## xxx
########################################################
@retry(exceptions=(Exception,), tries=3, delay=1, backoff=2)
async def process_single_frame(vlogger, video_id: str, frame_data: dict, s3_client, processed_frames_dir: str):
    @vlogger.log_performance
    async def _process_single_frame():
        try:
            frame_number = int(frame_data['frame_number'])
        except (KeyError, ValueError) as e:
            vlogger.logger.error(f"Invalid frame number in OCR results for video {video_id}: {str(e)}")
            return

        original_frame_key = f"{video_id}/frames/{frame_number:06d}.jpg"
        
        try:
            vlogger.logger.debug(f"Downloading frame {frame_number} for video {video_id}")
            frame_obj = await vlogger.log_performance(asyncio.to_thread)(
                s3_client.get_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=original_frame_key
            )
            frame_data = await frame_obj['Body'].read()
            vlogger.log_s3_operation("download", len(frame_data))
            
            frame_array = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        except Exception as e:
            vlogger.logger.error(f"Error downloading frame {frame_number} for video {video_id}: {str(e)}", exc_info=True)
            raise

        # Always annotate the frame, even if there are no detected brands
        vlogger.logger.debug(f"Annotating frame {frame_number} for video {video_id}")
        frame = annotate_frame(
            frame,
            frame_data.get('detected_brands', []),
            show_confidence=settings.SHOW_CONFIDENCE,
            text_bg_opacity=settings.TEXT_BG_OPACITY
        )

        processed_frame_path = os.path.join(processed_frames_dir, f"processed_frame_{frame_number:06d}.jpg")
        cv2.imwrite(processed_frame_path, frame)

        try:
            vlogger.logger.debug(f"Uploading processed frame {frame_number} for video {video_id}")
            with open(processed_frame_path, 'rb') as f:
                await vlogger.log_performance(s3_client.put_object)(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=f"{video_id}/processed_frames/processed_frame_{frame_number:06d}.jpg",
                    Body=f
                )
            vlogger.log_s3_operation("upload", os.path.getsize(processed_frame_path))
        except Exception as e:
            vlogger.logger.error(f"Error uploading processed frame {frame_number} for video {video_id}: {str(e)}", exc_info=True)
            raise

        vlogger.logger.debug(f"Completed processing frame {frame_number} for video {video_id}")

    await _process_single_frame()
########################################################

## Annotate individual video frames
########################################################
def annotate_frame(
    vlogger,
    frame: np.ndarray,
    detected_brands: List[Dict],
    show_confidence: bool = True,
    text_bg_opacity: float = 0.7,
) -> np.ndarray:
    @vlogger.log_performance
    def _annotate_frame():
        if not isinstance(frame, np.ndarray):
            vlogger.logger.error("Frame must be a numpy array")
            raise ValueError("Frame must be a numpy array")

        vlogger.logger.debug(f"Annotating frame with {len(detected_brands)} detected brands")

        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Load a nicer font (you may need to adjust the path)
        font_size = 20
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            vlogger.logger.debug("Loaded DejaVuSans-Bold font")
        except IOError:
            vlogger.logger.warning("Failed to load DejaVuSans-Bold font, using default")
            font = ImageFont.load_default()

        for brand in detected_brands:
            if 'bounding_box' not in brand or 'vertices' not in brand['bounding_box']:
                vlogger.logger.warning(f"Skipping brand without bounding box: {brand}")
                continue  # Skip brands without bounding boxes

            vertices = brand['bounding_box']['vertices']
            text = str(brand.get('text', '')).upper()  # Capitalize the brand name
            confidence = float(brand.get('confidence', 0.0))

            pts = np.array([(int(v['x']), int(v['y'])) for v in vertices], np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            color = get_color_by_confidence(confidence)

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

            # Prepare label
            label = f"{text}" + (f" ({confidence:.2f})" if show_confidence else "")
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

            vlogger.logger.debug(f"Annotated brand: {text} at position ({x_min}, {y_min}, {x_max}, {y_max})")

        vlogger.logger.debug("Frame annotation completed")
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return _annotate_frame()
########################################################

## Reconstruct video with annotations
########################################################
async def reconstruct_video(vlogger, video_id: str, temp_dir: str):
    @vlogger.log_performance
    async def _reconstruct_video():
        dual_log(vlogger, app_logger, 'info', f"Starting video reconstruction for video {video_id}")
        processed_frames_dir = os.path.join(temp_dir, "processed_frames")
        output_video_path = os.path.join(temp_dir, f"{video_id}_output.mp4")
        final_output_path = os.path.join(temp_dir, f"{video_id}_final.mp4")
        audio_path = os.path.join(temp_dir, f"{video_id}_audio.mp3")

        try:
            # Fetch processing stats
            try:
                vlogger.logger.debug(f"Fetching processing stats for video {video_id}")
                stats_obj = await vlogger.log_performance(asyncio.to_thread)(
                    s3_client.get_object,
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=f'{video_id}/processing_stats.json'
                )
                stats_data = await stats_obj['Body'].read()
                vlogger.log_s3_operation("download", len(stats_data))
                stats = json.loads(stats_data.decode('utf-8'))
                original_fps = stats['video']['video_fps']
                vlogger.logger.info(f"Original video frame rate: {original_fps} fps")
            except Exception as e:
                vlogger.logger.error(f"Error fetching processing stats for video {video_id}: {str(e)}", exc_info=True)
                original_fps = 30  # Fallback to 30 fps if we can't get the original
                vlogger.logger.warning(f"Using fallback frame rate of {original_fps} fps")

            frame_pattern = os.path.join(processed_frames_dir, 'processed_frame_%06d.jpg')
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
            vlogger.logger.info(f"Running FFmpeg command to create video for {video_id}")
            result = await asyncio.subprocess.create_subprocess_exec(
                *ffmpeg_command, 
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, ffmpeg_command, stderr)
            vlogger.logger.debug(f"FFmpeg output: {stdout.decode()}")

            vlogger.logger.info(f"Downloading audio file for video {video_id}")
            await vlogger.log_performance(s3_client.download_file)(
                settings.PROCESSING_BUCKET, 
                f"{video_id}/audio.mp3", 
                audio_path
            )
            vlogger.log_s3_operation("download", os.path.getsize(audio_path))

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
            vlogger.logger.info(f"Running FFmpeg command to combine video and audio for {video_id}")
            result = await asyncio.subprocess.create_subprocess_exec(
                *combine_command, 
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, combine_command, stderr)
            vlogger.logger.debug(f"FFmpeg combine output: {stdout.decode()}")

            dual_log(vlogger, app_logger, 'info', f"Uploading processed video for {video_id}")
            await vlogger.log_performance(s3_client.upload_file)(
                final_output_path, 
                settings.PROCESSING_BUCKET, 
                f"{video_id}/processed_video.mp4"
            )
            vlogger.log_s3_operation("upload", os.path.getsize(final_output_path))

            dual_log(vlogger, app_logger, 'info', f"Successfully processed and uploaded video {video_id}")

        except subprocess.CalledProcessError as e:
            dual_log(vlogger, app_logger, 'error', f"FFmpeg error for {video_id}: {e.stderr.decode()}", exc_info=True)
            raise
        except Exception as e:
            dual_log(vlogger, app_logger, 'error', f"Error in reconstruct_video for {video_id}: {str(e)}", exc_info=True)
            raise
        finally:
            for file_path in [output_video_path, final_output_path, audio_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        vlogger.logger.debug(f"Removed temporary file: {file_path}")
                except Exception as e:
                    vlogger.logger.warning(f"Error removing temporary file {file_path}: {str(e)}")

    await _reconstruct_video()
    dual_log(vlogger, app_logger, 'info', f"Completed video reconstruction for video {video_id}")
########################################################

## Get colour by confidence
########################################################
def get_color_by_confidence(vlogger, confidence: float) -> Tuple[int, int, int]:
    @vlogger.log_performance
    def _get_color_by_confidence():
        # Ensure confidence is between 0 and 1
        confidence = max(0, min(1, confidence))
        
        vlogger.logger.debug(f"Calculating color for confidence: {confidence}")

        # Define color ranges
        if confidence < 0.5:
            # Red (255, 0, 0) to Yellow (255, 255, 0)
            r = 255
            g = int(255 * (confidence * 2))
            b = 0
            vlogger.logger.debug(f"Confidence < 0.5, using red to yellow range. Color: ({r}, {g}, {b})")
        else:
            # Yellow (255, 255, 0) to Green (0, 255, 0)
            r = int(255 * ((1 - confidence) * 2))
            g = 255
            b = 0
            vlogger.logger.debug(f"Confidence >= 0.5, using yellow to green range. Color: ({r}, {g}, {b})")

        return (r, g, b)

    return _get_color_by_confidence()
########################################################