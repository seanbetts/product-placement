import cv2
import numpy as np
import json
import os
import tempfile
from io import BytesIO
from typing import List, Dict
from moviepy.editor import VideoFileClip, AudioFileClip
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client

def process_video_frames(video_id: str):
    s3_client = get_s3_client()
    
    # Load OCR results
    ocr_results_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f'{video_id}/ocr/brands_ocr.json')
    ocr_results = json.loads(ocr_results_obj['Body'].read().decode('utf-8'))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_frames_dir = os.path.join(temp_dir, "processed_frames")
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        # Process each frame
        for frame_data in ocr_results:
            frame_number = frame_data['frame_number']
            original_frame_key = f"{video_id}/frames/{frame_number:06d}.jpg"
            
            # Download the original frame
            frame_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=original_frame_key)
            frame_array = np.frombuffer(frame_obj['Body'].read(), np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            # Annotate the frame if brands are detected
            if frame_data['detected_brands']:
                frame = annotate_frame(frame, frame_data['detected_brands'])
            
            # Save the processed frame locally
            processed_frame_path = os.path.join(processed_frames_dir, f"processed_frame_{frame_number:04d}.jpg")
            cv2.imwrite(processed_frame_path, frame)
            
            # Upload the processed frame to S3
            with open(processed_frame_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=f"{video_id}/processed_frames/processed_frame_{frame_number:06d}.jpg",
                    Body=f
                )
        
        # Reconstruct video from processed frames
        reconstruct_video(video_id, temp_dir)

def annotate_frame(frame: np.ndarray, detected_brands: List[Dict]) -> np.ndarray:
    for brand in detected_brands:
        if 'bounding_box' in brand:
            vertices = brand['bounding_box']['vertices']
            text = brand['text']
            confidence = brand['confidence']
            
            # Draw bounding box
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # Add text label
            label = f"{text} ({confidence:.2f})"
            cv2.putText(frame, label, (vertices[0]['x'], vertices[0]['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def reconstruct_video(video_id: str, temp_dir: str):
    s3_client = get_s3_client()
    processed_frames_dir = os.path.join(temp_dir, "processed_frames")
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")
    
    # Get the list of processed frames from S3
    response = s3_client.list_objects_v2(
        Bucket=settings.PROCESSING_BUCKET,
        Prefix=f"{video_id}/processed_frames/"
    )
    frame_files = sorted([obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.jpg')])
    
    # Download the first frame to get dimensions
    first_frame_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=frame_files[0])
    first_frame_array = np.frombuffer(first_frame_obj['Body'].read(), np.uint8)
    first_frame = cv2.imdecode(first_frame_array, cv2.IMREAD_COLOR)
    height, width, _ = first_frame.shape
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    
    # Write frames to video
    for frame_key in frame_files:
        frame_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=frame_key)
        frame_array = np.frombuffer(frame_obj['Body'].read(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        out.write(frame)
    
    out.release()
    
    # Download audio file
    audio_obj = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=f"{video_id}/audio.mp3")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    with open(audio_path, 'wb') as f:
        f.write(audio_obj['Body'].read())
    
    # Add audio to the video
    video = VideoFileClip(output_video_path)
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_output_path = os.path.join(temp_dir, "final_processed_video.mp4")
    final_video.write_videofile(final_output_path, codec='libx264')
    
    # Upload the final video to S3
    with open(final_output_path, 'rb') as f:
        s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=f"{video_id}/processed_video.mp4",
            Body=f
        )

# Main function to process video
def process_video(video_id: str):
    process_video_frames(video_id)
    print(f"Video processing completed for video_id: {video_id}")