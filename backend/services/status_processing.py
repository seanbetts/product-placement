import json
import datetime
import asyncio
from typing import Dict, Any
from fastapi import HTTPException
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from services import video_processing
from botocore.exceptions import ClientError

## Get processing status
########################################################
async def get_processing_status(video_id: str) -> Dict[str, Any]:
    # logger.info(f"Received status request for video ID: {video_id}")
    status_key = f'{video_id}/status.json'
    s3_client = get_s3_client()
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=status_key)
        status_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # logger.info(f"Status for video ID {video_id}: {status_data}")
        return status_data
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"Status not found for video ID: {video_id}")
            raise HTTPException(status_code=404, detail="Video status not found")
        else:
            logger.error(f"Error retrieving status for video ID {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving video status")
########################################################

## Update processing status
########################################################
async def periodic_status_update(video_id: str, status_tracker: StatusTracker, s3_client):
    while True:
        status_tracker.calculate_overall_progress()
        status_tracker.update_s3_status(s3_client)
        await asyncio.sleep(settings.STATUS_UPDATE_INTERVAL)
########################################################

## Mark video as completed processing
########################################################
def mark_video_as_completed(video_id: str):
    # Update the processing_stats.json to mark it as completed
    stats_key = f'{video_id}/processing_stats.json'
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=stats_key)
        stats_data = json.loads(response['Body'].read().decode('utf-8'))
        stats_data['status'] = 'completed'
        stats_data['completion_time'] = datetime.datetime.utcnow().isoformat()
        
        s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=stats_key,
            Body=json.dumps(stats_data),
            ContentType='application/json'
        )
        
        # Update the completed videos list
        video_processing.update_completed_videos_list(video_id)
        
        # logger.info(f"Marked video {video_id} as completed")
    except Exception as e:
        logger.error(f"Error marking video {video_id} as completed: {str(e)}")
########################################################

## Gets processing stats for a video
########################################################
async def get_processing_stats(video_id: str):
    # logger.info(f"Received request for processing stats of video: {video_id}") 
    stats_key = f'{video_id}/processing_stats.json'
    s3_client = get_s3_client()
    
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=stats_key)
        stats = json.loads(response['Body'].read().decode('utf-8'))
        return stats
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Processing stats not found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Processing stats not found")
    except Exception as e:
        logger.error(f"Error retrieving processing stats for video ID {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving processing stats")
########################################################

## Gets list of processed videos
########################################################
async def get_processed_videos():
    completed_videos_key = '_completed_videos.json'
    s3_client = get_s3_client()
    try:
        # Get the list of completed video IDs
        response = await asyncio.to_thread(
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET,
            Key=completed_videos_key
        )
        completed_video_ids = json.loads(await asyncio.to_thread(response['Body'].read))
        
        processed_videos = []
        for video_id in completed_video_ids:
            try:
                # Get the processing_stats.json for this video
                stats_key = f'{video_id}/processing_stats.json'
                stats_response = await asyncio.to_thread(
                    s3_client.get_object,
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=stats_key
                )
                stats_data = json.loads(await asyncio.to_thread(stats_response['Body'].read))
                processed_videos.append({
                    'video_id': video_id,
                    'details': stats_data
                })
            except Exception as e:
                logger.error(f"Error retrieving details for video {video_id}: {str(e)}")
        
        logger.info(f"Returning {len(processed_videos)} processed videos")
        return processed_videos
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info("No completed videos found")
            return []
        else:
            logger.error(f"Error retrieving completed videos list: {str(e)}")
            raise
########################################################