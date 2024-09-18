import cv2
import numpy as np
import json
import os
import tempfile
import subprocess
from core.config import settings
from retry import retry
from typing import List, Dict, Tuple
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
        logger.info(f"Completed video processing for video {video_id}")

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
            annotation_style=settings.ANNOTATION_STYLE,
            text_bg_opacity=settings.TEXT_BG_OPACITY,
            rounded_corners=settings.ROUNDED_CORNERS
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
    annotation_style: str = "box",
    text_bg_opacity: float = 0.7,
    rounded_corners: bool = True
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy array")
    
    annotated_frame = frame.copy()

    for brand in detected_brands:
        if 'bounding_box' not in brand or 'vertices' not in brand['bounding_box']:
            continue

        vertices = brand['bounding_box']['vertices']
        text = str(brand.get('text', ''))
        confidence = float(brand.get('confidence', 0.0))

        pts = np.array([(int(v['x']), int(v['y'])) for v in vertices], np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        color = get_color_by_confidence(confidence)
        
        try:
            if annotation_style == "box":
                draw_rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2, rounded_corners)
            elif annotation_style == "underline":
                cv2.line(annotated_frame, (x_min, y_max), (x_max, y_max), color, 2)
            
            font_scale = min((x_max - x_min) / 200, 1.0)
            label = f"{text}" + (f" ({confidence:.2f})" if show_confidence else "")
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_x = x_min
            text_y = y_min - 10 if y_min > text_height + 10 else y_max + text_height + 10
            
            bg_rect = annotated_frame[text_y-text_height-5:text_y+5, text_x:text_x+text_width+10]
            overlay = bg_rect.copy()
            cv2.rectangle(overlay, (0, 0), (text_width+10, text_height+10), color, -1)
            cv2.addWeighted(overlay, text_bg_opacity, bg_rect, 1 - text_bg_opacity, 0, bg_rect)
            annotated_frame[text_y-text_height-5:text_y+5, text_x:text_x+text_width+10] = bg_rect
            
            cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Error annotating brand: {e}")
            continue

    return annotated_frame

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
            logger.info(f"Original video frame rate: {original_fps} fps")
        except Exception as e:
            logger.error(f"Error fetching processing stats for video {video_id}: {str(e)}")
            original_fps = 30  # Fallback to 30 fps if we can't get the original
            logger.warning(f"Using fallback frame rate of {original_fps} fps")

        response = s3_client.list_objects_v2(
            Bucket=settings.PROCESSING_BUCKET,
            Prefix=f"{video_id}/processed_frames/"
        )
        
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
        
        subprocess.run(ffmpeg_command, check=True)
        
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
        
        subprocess.run(combine_command, check=True)
        
        s3_client.upload_file(final_output_path, settings.PROCESSING_BUCKET, f"{video_id}/processed_video.mp4")
        
        logger.info(f"Successfully processed and uploaded video {video_id}")
        
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

@retry(exceptions=(IOError, subprocess.CalledProcessError), tries=3, delay=1, backoff=2)
def write_video_with_retry(video_clip, output_path, codec='libx264', audio_codec='aac'):
    temp_audio_path = tempfile.mktemp(suffix=".aac", dir=TEMP_DIR)
    try:
        video_clip.write_videofile(output_path, codec=codec, audio_codec=audio_codec, temp_audiofile=temp_audio_path)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def get_color_by_confidence(confidence: float) -> Tuple[int, int, int]:
    if confidence > 0.8:
        return (0, 255, 0)  # Green for high confidence
    elif confidence > 0.5:
        return (0, 255, 255)  # Yellow for medium confidence
    else:
        return (0, 0, 255)  # Red for low confidence

def draw_rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int, rounded: bool = True) -> None:
    try:
        if rounded:
            radius = min(10, (pt2[0] - pt1[0]) // 4, (pt2[1] - pt1[1]) // 4)
            cv2.line(img, (pt1[0] + radius, pt1[1]), (pt2[0] - radius, pt1[1]), color, thickness)
            cv2.line(img, (pt1[0] + radius, pt2[1]), (pt2[0] - radius, pt2[1]), color, thickness)
            cv2.line(img, (pt1[0], pt1[1] + radius), (pt1[0], pt2[1] - radius), color, thickness)
            cv2.line(img, (pt2[0], pt1[1] + radius), (pt2[0], pt2[1] - radius), color, thickness)
            cv2.ellipse(img, (pt1[0] + radius, pt1[1] + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (pt2[0] - radius, pt1[1] + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (pt1[0] + radius, pt2[1] - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (pt2[0] - radius, pt2[1] - radius), (radius, radius), 0, 0, 90, color, thickness)
        else:
            cv2.rectangle(img, pt1, pt2, color, thickness)
    except cv2.error:
        cv2.rectangle(img, pt1, pt2, color, thickness)