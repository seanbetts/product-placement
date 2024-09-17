import cv2
import numpy as np
import json
import os
from typing import List, Dict
from moviepy.editor import VideoFileClip, AudioFileClip
from core.config import settings

def process_video_frames(video_id: str):
    processing_bucket = settings.PROCESSING_BUCKET
    # Load OCR results
    ocr_results_path = f"{processing_bucket}/{video_id}/ocr/brands_ocr.json"
    with open(ocr_results_path, 'r') as f:
        ocr_results = json.load(f)

    # Create processed_frames directory
    processed_frames_dir = f"{processing_bucket}/{video_id}/processed_frames"
    os.makedirs(processed_frames_dir, exist_ok=True)

    # Process each frame
    for frame_data in ocr_results:
        frame_number = frame_data['frame_number']
        original_frame_path = f"{processing_bucket}/{video_id}/frames/frame_{frame_number:04d}.jpg"
        
        # Load the original frame
        frame = cv2.imread(original_frame_path)

        # Annotate the frame if brands are detected
        if frame_data['detected_brands']:
            frame = annotate_frame(frame, frame_data['detected_brands'])

        # Save the processed frame
        processed_frame_path = f"{processed_frames_dir}/processed_frame_{frame_number:04d}.jpg"
        cv2.imwrite(processed_frame_path, frame)

    # Reconstruct video from processed frames
    reconstruct_video(video_id, processing_bucket)

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

def reconstruct_video(video_id: str):
    processing_bucket = settings.PROCESSING_BUCKET

    processed_frames_dir = f"{processing_bucket}/{video_id}/processed_frames"
    output_video_path = f"{processing_bucket}/{video_id}/processed_video.mp4"
    audio_path = f"{processing_bucket}/{video_id}/audio.mp3"

    # Get the list of processed frames
    frame_files = sorted([f for f in os.listdir(processed_frames_dir) if f.endswith('.jpg')])

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(processed_frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(processed_frames_dir, frame_file))
        out.write(frame)

    out.release()

    # Add audio to the video
    video = VideoFileClip(output_video_path)
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_video_path, codec='libx264')

# Main function to process video
def process_video(video_id: str):
    processing_bucket = settings.PROCESSING_BUCKET

    process_video_frames(video_id, processing_bucket)
    print(f"Video processing completed for video_id: {video_id}")