import asyncio
import boto3
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
async def detect_objects(video_id: str):
    try:
        status_tracker = StatusTracker(video_id)
        video_details = VideoDetails(video_id)
        
        logger.info(f"Starting object detection for video {video_id}")
        
        total_frames = 2080
        
        raw_results = {}
        for frame_number in range(total_frames):
            try:
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
                                'LabelInclusionFilters': [
                                    'Can', 'Pizza', 'Sneaker', 'Cap', 'Pill'
                                ]
                            }
                        }
                    )
                )
                
                frame_key = f'{frame_number:06d}.jpg'
                raw_results[frame_key] = rekognition_response.get('Labels', [])
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Frame {frame_number:06d}.jpg not found for video {video_id}")
                else:
                    logger.info(f"Error processing frame {frame_number:06d}.jpg for video {video_id}: {str(e)}")
            except Exception as e:
                logger.info(f"Error processing frame {frame_number:06d}.jpg for video {video_id}: {str(e)}")
        
        # Save combined raw results to S3
        await s3_operations.save_data_to_s3(video_id, 'raw_object_detection_results.json', raw_results)
        
        logger.info(f"Object detection completed for video {video_id}")
        return raw_results
        
    except Exception as e:
        logger.info(f"Error in object detection for video {video_id}: {str(e)}")
        raise
########################################################