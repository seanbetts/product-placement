import cv2
import numpy as np
import json
import os
import tempfile
import subprocess
from core.config import settings
from retry import retry
from typing import List, Dict, Tuple
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client

TEMP_DIR = settings.TEMP_DIR

def process_video_frames(video_id: str):
    s3_client = get_s3_client()
    
    try:
        # Load OCR results
        ocr_results_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/ocr/brands_ocr.json')
        ocr_results = json.loads(ocr_results_obj['Body'].read().decode('utf-8'))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_frames_dir = os.path.join(temp_dir, "processed_frames")
            os.makedirs(processed_frames_dir, exist_ok=True)
            
            # Process each frame
            for frame_data in ocr_results:
                try:
                    frame_number = int(frame_data['frame_number'])
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid frame number in OCR results for video {video_id}: {str(e)}")
                    continue
                
                original_frame_key = f"{video_id}/frames/{frame_number:06d}.jpg"
                
                # Download the original frame
                try:
                    frame_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=original_frame_key)
                    frame_array = np.frombuffer(frame_obj['Body'].read(), np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                except Exception as e:
                    logger.error(f"Error downloading frame {frame_number} for video {video_id}: {str(e)}")
                    continue
                
                # Annotate the frame if brands are detected
                if frame_data.get('detected_brands'):
                    frame = annotate_frame(
                        frame, 
                        frame_data['detected_brands'],
                        show_confidence=False,
                        annotation_style="box",
                        text_bg_opacity=0.7,
                        rounded_corners=True
                    )
                
                # Save the processed frame locally
                processed_frame_path = os.path.join(processed_frames_dir, f"processed_frame_{frame_number:06d}.jpg")
                cv2.imwrite(processed_frame_path, frame)
                
                # Upload the processed frame to S3
                try:
                    with open(processed_frame_path, 'rb') as f:
                        s3_client.put_object(
                            Bucket=settings.PROCESSING_BUCKET,
                            Key=f"{video_id}/processed_frames/processed_frame_{frame_number:06d}.jpg",
                            Body=f
                        )
                except Exception as e:
                    logger.error(f"Error uploading processed frame {frame_number} for video {video_id}: {str(e)}")
            
            # Reconstruct video from processed frames
            reconstruct_video(video_id, temp_dir)
    except Exception as e:
        logger.error(f"Error in process_video_frames for video {video_id}: {str(e)}", exc_info=True)
        raise

def annotate_frame(
    frame: np.ndarray,
    detected_brands: List[Dict],
    show_confidence: bool = True,
    annotation_style: str = "box",
    text_bg_opacity: float = 0.7,
    rounded_corners: bool = True
) -> np.ndarray:
    for brand in detected_brands:
        if 'bounding_box' not in brand or 'vertices' not in brand['bounding_box']:
            continue

        vertices = brand['bounding_box']['vertices']
        text = str(brand.get('text', ''))
        confidence = float(brand.get('confidence', 0.0))

        # Ensure vertices are integers
        pts = np.array([(int(v['x']), int(v['y'])) for v in vertices], np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        color = get_color_by_confidence(confidence)
        
        if annotation_style == "box":
            if rounded_corners:
                draw_rounded_rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2, 10)
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        elif annotation_style == "underline":
            cv2.line(frame, (x_min, y_max), (x_max, y_max), color, 2)
        
        # Calculate text size and position
        font_scale = min((x_max - x_min) / 200, 1.0)  # Adjust text size based on box width
        label = f"{text}" + (f" ({confidence:.2f})" if show_confidence else "")
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_x = x_min
        text_y = y_min - 10 if y_min > text_height + 10 else y_max + text_height + 10
        
        # Draw semi-transparent background for text
        bg_rect = frame[text_y-text_height-5:text_y+5, text_x:text_x+text_width+10]
        overlay = bg_rect.copy()
        cv2.rectangle(overlay, (0, 0), (text_width+10, text_height+10), color, -1)
        cv2.addWeighted(overlay, text_bg_opacity, bg_rect, 1 - text_bg_opacity, 0, bg_rect)
        frame[text_y-text_height-5:text_y+5, text_x:text_x+text_width+10] = bg_rect
        
        # Add text label
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return frame

def reconstruct_video(video_id: str, temp_dir: str):
    s3_client = get_s3_client()
    processed_frames_dir = os.path.join(temp_dir, "processed_frames")
    
    # Use TEMP_DIR for output files
    output_video_path = os.path.join(TEMP_DIR, f"{video_id}_output.mp4")
    final_output_path = os.path.join(TEMP_DIR, f"{video_id}_final.mp4")
    audio_path = os.path.join(TEMP_DIR, f"{video_id}_audio.mp3")
    
    try:
        # Get the list of processed frames from S3
        response = s3_client.list_objects_v2(
            Bucket=settings.PROCESSING_BUCKET,
            Prefix=f"{video_id}/processed_frames/"
        )
        
        # Use ffmpeg to create video from frames
        frame_pattern = os.path.join(processed_frames_dir, 'processed_frame_%06d.jpg')
        ffmpeg_command = [
            'ffmpeg',
            '-framerate', '30',
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]
        
        subprocess.run(ffmpeg_command, check=True)
        
        # Download audio file
        s3_client.download_file(settings.PROCESSING_BUCKET, f"{video_id}/audio.mp3", audio_path)
        
        # Combine video and audio using ffmpeg
        combine_command = [
            'ffmpeg',
            '-i', output_video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            final_output_path
        ]
        
        subprocess.run(combine_command, check=True)
        
        # Upload the final video to S3
        s3_client.upload_file(final_output_path, settings.PROCESSING_BUCKET, f"{video_id}/processed_video.mp4")
        
        logger.info(f"Successfully processed and uploaded video {video_id}")
        
    except Exception as e:
        logger.error(f"Error in reconstruct_video for {video_id}: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up temporary files
        logger.info("Cleaning up temporary files")
        for file_path in [output_video_path, final_output_path, audio_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {file_path}: {str(e)}")


@retry(exceptions=(IOError, subprocess.CalledProcessError), tries=3, delay=1, backoff=2)
def write_video_with_retry(video_clip, output_path, codec='libx264', audio_codec='aac'):
    temp_audio_path = tempfile.mktemp(suffix=".aac", dir=TEMP_DIR)
    try:
        video_clip.write_videofile(output_path, codec=codec, audio_codec=audio_codec, temp_audiofile=temp_audio_path)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def get_color_by_confidence(confidence: float) -> Tuple[int, int, int]:
    """Return a color based on the confidence score."""
    if confidence > 0.8:
        return (0, 255, 0)  # Green for high confidence
    elif confidence > 0.5:
        return (0, 255, 255)  # Yellow for medium confidence
    else:
        return (0, 0, 255)  # Red for low confidence

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius):
    """Draw a rounded rectangle on the image."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangle
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    
    # Draw corner circles
    cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)