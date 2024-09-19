import cv2
import numpy as np
import json
import os
import tempfile
import subprocess
from PIL import Image, ImageDraw, ImageFont
from retry import retry
from typing import List, Dict, Tuple
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from concurrent.futures import ThreadPoolExecutor, as_completed

TEMP_DIR = settings.TEMP_DIR

def process_video_frames(video_id: str):
    logger.info(f"Starting post-processing for video {video_id}")
    s3_client = get_s3_client()
    
    try:
        # Load OCR results
        ocr_results_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/ocr/brands_ocr.json')
        ocr_results = json.loads(ocr_results_obj['Body'].read().decode('utf-8'))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_frames_dir = os.path.join(temp_dir, "processed_frames")
            os.makedirs(processed_frames_dir, exist_ok=True)
            
            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
                futures = [executor.submit(process_single_frame, video_id, frame_data, s3_client, processed_frames_dir) 
                           for frame_data in ocr_results]
                
                total_frames = len(futures)
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        future.result()
                        # logger.info(f"Processed {i}/{total_frames} frames for video {video_id}")
                    except Exception as e:
                        logger.error(f"Error processing frame for video {video_id}: {str(e)}")
            
            # Reconstruct video from processed frames
            reconstruct_video(video_id, temp_dir)
    except Exception as e:
        logger.error(f"Error in process_video_frames for video {video_id}: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f"Completed video post-processing for video {video_id}")

@retry(exceptions=(Exception,), tries=3, delay=1, backoff=2)
def process_single_frame(video_id: str, frame_data: dict, s3_client, processed_frames_dir: str):
    try:
        frame_number = int(frame_data['frame_number'])
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid frame number in OCR results for video {video_id}: {str(e)}")
        return
    
    # logger.info(f"Processing frame {frame_number} for video {video_id}")
    
    original_frame_key = f"{video_id}/frames/{frame_number:06d}.jpg"
    
    try:
        frame_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=original_frame_key)
        frame_array = np.frombuffer(frame_obj['Body'].read(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error downloading frame {frame_number} for video {video_id}: {str(e)}")
        raise
    
    if frame_data.get('detected_brands'):
        frame = annotate_frame(
            frame, 
            frame_data['detected_brands'],
            show_confidence=settings.SHOW_CONFIDENCE,
            text_bg_opacity=settings.TEXT_BG_OPACITY
        )
    
    processed_frame_path = os.path.join(processed_frames_dir, f"processed_frame_{frame_number:06d}.jpg")
    cv2.imwrite(processed_frame_path, frame)
    
    try:
        with open(processed_frame_path, 'rb') as f:
            s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f"{video_id}/processed_frames/processed_frame_{frame_number:06d}.jpg",
                Body=f
            )
    except Exception as e:
        logger.error(f"Error uploading processed frame {frame_number} for video {video_id}: {str(e)}")
        raise
    
    # logger.info(f"Completed processing frame {frame_number} for video {video_id}")

def annotate_frame(
    frame: np.ndarray,
    detected_brands: List[Dict],
    show_confidence: bool = True,
    text_bg_opacity: float = 0.7,
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy array")
    
    # Convert OpenCV BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Load a nicer font (you may need to adjust the path)
    font_size = 20
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for brand in detected_brands:
        if 'bounding_box' not in brand or 'vertices' not in brand['bounding_box']:
            continue

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

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def reconstruct_video(video_id: str, temp_dir: str):
    logger.info(f"Starting video reconstruction for video {video_id}")
    s3_client = get_s3_client()
    processed_frames_dir = os.path.join(temp_dir, "processed_frames")
    
    output_video_path = os.path.join(TEMP_DIR, f"{video_id}_output.mp4")
    final_output_path = os.path.join(TEMP_DIR, f"{video_id}_final.mp4")
    audio_path = os.path.join(TEMP_DIR, f"{video_id}_audio.mp3")
    
    try:
        # Fetch processing stats
        try:
            stats_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/processing_stats.json')
            stats = json.loads(stats_obj['Body'].read().decode('utf-8'))
            original_fps = stats['video']['video_fps']
            # logger.info(f"Original video frame rate: {original_fps} fps")
        except Exception as e:
            logger.error(f"Error fetching processing stats for video {video_id}: {str(e)}")
            original_fps = 30  # Fallback to 30 fps if we can't get the original
            logger.warning(f"Using fallback frame rate of {original_fps} fps")

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
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        # logger.info(f"FFmpeg output: {result.stdout}")
        
        s3_client.download_file(settings.PROCESSING_BUCKET, f"{video_id}/audio.mp3", audio_path)
        
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
        result = subprocess.run(combine_command, check=True, capture_output=True, text=True)
        # logger.info(f"FFmpeg combine output: {result.stdout}")
        
        s3_client.upload_file(final_output_path, settings.PROCESSING_BUCKET, f"{video_id}/processed_video.mp4")
        
        logger.info(f"Successfully processed and uploaded video {video_id}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for {video_id}: {e.output}")
        raise
    except Exception as e:
        logger.error(f"Error in reconstruct_video for {video_id}: {str(e)}", exc_info=True)
        raise
    finally:
        for file_path in [output_video_path, final_output_path, audio_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {file_path}: {str(e)}")
        logger.info(f"Completed video reconstruction for video {video_id}")

def get_color_by_confidence(confidence: float) -> Tuple[int, int, int]:
    if confidence > 0.8:
        return (0, 255, 0)  # Green for high confidence
    elif confidence > 0.5:
        return (0, 255, 255)  # Yellow for medium confidence
    else:
        return (0, 0, 255)  # Red for low confidence