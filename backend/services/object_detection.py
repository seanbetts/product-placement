import json
import asyncio
import boto3
from typing import Dict, Any
from core.config import settings
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from core.logging import logger
from services import s3_operations
from botocore.exceptions import ClientError

# Initialize Rekognition client
rekognition_client = boto3.client('rekognition', region_name=settings.AWS_DEFAULT_REGION)

## Detect objects in video
########################################################
async def detect_objects(video_id: str, status_tracker: 'StatusTracker', video_details: VideoDetails, brand_results: list):
    try:
        # Set initial status
        await status_tracker.update_process_status("objects", "processing", 0)

        # Load brand database
        with open(settings.BRAND_DATABASE, 'r') as f:
            brand_database = json.load(f)

        total_frames = video_details("number_of_frames")
        raw_results = {}

        for frame_number in range(total_frames):
            try:
                # Get brands detected in this frame
                frame_brands = [brand['text'] for brand in next((item['detected_brands'] for item in brand_results if item['frame_number'] == frame_number), [])]

                # Get objects for detected brands
                objects_to_detect = set()
                for brand in frame_brands:
                    brand_lower = brand.lower()
                    for db_brand, brand_info in brand_database.items():
                        if brand_lower == db_brand or brand_lower in brand_info['variations']:
                            objects_to_detect.update(brand_info['objects'])

                # If no brands detected, use a default set of objects
                if not objects_to_detect:
                    objects_to_detect = {'Can', 'Pizza', 'Sneaker', 'Cap', 'Pill'}

                s3_object = {
                    'Bucket': settings.PROCESSING_BUCKET,
                    'Name': f'{video_id}/frames/{frame_number:06d}.jpg'
                }
                loop = asyncio.get_running_loop()
                rekognition_response = await loop.run_in_executor(
                    None,
                    lambda: rekognition_client.detect_labels(
                        Image={'S3Object': s3_object},
                        Features=['GENERAL_LABELS'],
                        Settings={
                            'GeneralLabels': {
                                'LabelInclusionFilters': list(objects_to_detect)
                            }
                        }
                    )
                )
                frame_key = f'{frame_number:06d}.jpg'
                raw_results[frame_key] = rekognition_response.get('Labels', [])

                # Update progress
                progress = (frame_number + 1) / total_frames * 100
                await status_tracker.update_process_status("objects", "processing", progress)

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.error(f"Video Processing - Object Detection - Step 4.1: Frame {frame_number:06d}.jpg not found for video {video_id}")
                else:
                    logger.error(f"Video Processing - Object Detection - Step 4.1: Error processing frame {frame_number:06d}.jpg for video {video_id}: {str(e)}")
            except Exception as e:
                logger.error(f"Video Processing - Object Detection - Step 4.1: Error processing frame {frame_number:06d}.jpg for video {video_id}: {str(e)}")

        # Save combined raw results to S3
        await s3_operations.save_data_to_s3(video_id, 'raw_object_detection_results.json', raw_results)
        logger.info(f"Video Processing - Object Detection - Step 4.2: Object detection completed for video {video_id}")

        # Set status to complete
        await status_tracker.update_process_status("objects", "complete", 100)

        logger.info(f"Video Processing - Object Detection - Step 4.3: Combining brand and object stats for annotation: {video_id}")
        annotation_objects = await combine_object_and_brand_data(video_id, status_tracker, video_details, brand_results, raw_results)

        return annotation_objects

    except Exception as e:
        logger.error(f"Video Processing - Object Detection: Error in object detection for video {video_id}: {str(e)}")
        # Set status to error
        await status_tracker.update_process_status("objects", "error")
        raise
########################################################


## Combine brand and object stats
########################################################
async def combine_object_and_brand_data(video_id: str, status_tracker: 'StatusTracker', video_details: VideoDetails, brand_results: list, raw_results: Dict[str, Any]):
    try:
        # Set initial status
        await status_tracker.update_process_status("annotation", "processing", 0)

        # Get video resolution
        width, height = video_details("resolution")

        # Process each frame
        annotation_objects = []
        total_frames = len(raw_results)

        for frame_number, frame_data in raw_results.items():
            frame_number = int(frame_number.split('.')[0])  # Convert '000503.jpg' to 503
            
            # Get brand data for this frame
            frame_brands = next((item['detected_brands'] for item in brand_results if item['frame_number'] == frame_number), [])

            frame_objects = []
            for obj in frame_data:
                # Only include objects with bounding boxes
                if obj['Instances']:
                    for instance in obj['Instances']:
                        bbox = instance['BoundingBox']
                        # Convert bounding box to vertices
                        vertices = [
                            {"x": int(bbox['Left'] * width), "y": int(bbox['Top'] * height)},
                            {"x": int((bbox['Left'] + bbox['Width']) * width), "y": int(bbox['Top'] * height)},
                            {"x": int((bbox['Left'] + bbox['Width']) * width), "y": int((bbox['Top'] + bbox['Height']) * height)},
                            {"x": int(bbox['Left'] * width), "y": int((bbox['Top'] + bbox['Height']) * height)}
                        ]
                        frame_objects.append({
                            "name": obj['Name'],
                            "confidence": instance['Confidence'],
                            "bounding_box": {
                                "vertices": vertices
                            }
                        })

            annotation_objects.append({
                "frame_number": frame_number,
                "detected_brands": frame_brands,
                "detected_objects": frame_objects
            })

            # Update progress
            progress = (frame_number + 1) / total_frames * 100
            await status_tracker.update_process_status("annotation", "processing", progress)

        # Save combined results to S3
        await s3_operations.save_data_to_s3(video_id, 'annotation_objects.json', annotation_objects)
        logger.info(f"Video Processing - Annotation: Object and brand annotation completed for video {video_id}")

        # Set status to complete
        await status_tracker.update_process_status("annotation", "complete", 100)

        return annotation_objects

    except Exception as e:
        logger.error(f"Video Processing - Annotation: Error in combining object and brand data for video {video_id}: {str(e)}")
        # Set status to error
        await status_tracker.update_process_status("annotation", "error")
        raise
########################################################