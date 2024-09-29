import json
import datetime
import asyncio
from typing import Dict, Any
from fastapi import HTTPException
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from services import video_processing
from botocore.exceptions import ClientError

## Get processing status
########################################################
async def get_processing_status(video_id: str) -> Dict[str, Any]:
    logger.debug(f"Received status request for video ID: {video_id}")
    status_key = f'{video_id}/status.json'
    
    try:
        logger.debug(f"Attempting to retrieve status for video ID: {video_id}")
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=status_key
            )
        data = await response['Body'].read()
        status_data = json.loads(data.decode('utf-8'))
        logger.debug(f"Status for video ID {video_id}: {status_data}")
        return status_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"Status not found for video ID: {video_id}")
            raise HTTPException(status_code=404, detail="Video status not found")
        else:
            logger.error(f"Error retrieving status for video ID {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error retrieving video status")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding status JSON for video ID {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error decoding video status")
    except Exception as e:
        logger.error(f"Unexpected error retrieving status for video ID {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error retrieving video status")
########################################################

## Update processing status
########################################################
async def periodic_status_update(video_id: str, status_tracker: StatusTracker):
    async def update_status():
        try:
            status_tracker.calculate_overall_progress()
            await status_tracker.update_s3_status()
            logger.debug(f"Updated status for video {video_id}: {status_tracker.status}")
        except Exception as e:
            logger.error(f"Error updating status for video {video_id}: {str(e)}", exc_info=True)
            await status_tracker.set_error(f"Status update error: {str(e)}")

    logger.info(f"Starting periodic status updates for video {video_id}")
    update_count = 0
    try:
        while True:
            logger.debug(f"Initiating status update {update_count + 1} for video {video_id}")
            await update_status()
            update_count += 1
            logger.debug(f"Completed status update {update_count} for video {video_id}")
            await asyncio.sleep(settings.STATUS_UPDATE_INTERVAL)
    except asyncio.CancelledError:
        logger.debug(f"Periodic status updates stopped for video {video_id} after {update_count} updates")
    except Exception as e:
        logger.error(f"Unexpected error in periodic status updates for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.set_error(f"Unexpected error in status updates: {str(e)}")
    finally:
        # Perform a final status update
        try:
            await update_status()
            logger.debug(f"Final status update completed for video {video_id}")
        except Exception as e:
            logger.error(f"Error during final status update for video {video_id}: {str(e)}", exc_info=True)
########################################################

## Mark video as completed processing
########################################################
async def mark_video_as_completed(video_id: str):
    stats_key = f'{video_id}/processing_stats.json'
    
    try:
        # Get the current processing stats
        logger.debug(f"Retrieving processing stats for video {video_id}")
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=stats_key
            )
        data = await response['Body'].read()
        stats_data = json.loads(data.decode('utf-8'))

        # Update the stats
        stats_data['status'] = 'completed'
        stats_data['completion_time'] = datetime.datetime.utcnow().isoformat()

        # Write the updated stats back to S3
        logger.debug(f"Updating processing stats for video {video_id}")
        updated_stats_json = json.dumps(stats_data)
        await s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=stats_key,
            Body=updated_stats_json,
            ContentType='application/json'
        )

        # Update the completed videos list
        logger.debug(f"Updating completed videos list for video {video_id}")
        await video_processing.update_completed_videos_list(video_id)

        logger.debug(f"Marked video {video_id} as completed")

    except Exception as e:
        logger.error(f"Error marking video {video_id} as completed: {str(e)}", exc_info=True)
        # Optionally, you might want to re-raise the exception here, depending on how you want to handle errors
########################################################

## Gets processing stats for a video
########################################################
async def get_processing_stats(video_id: str):
    logger.debug(f"Received request for processing stats of video: {video_id}")
    stats_key = f'{video_id}/processing_stats.json'
    
    try:
        logger.debug(f"Attempting to retrieve processing stats for video ID: {video_id}")
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=stats_key
            )
        data = await response['Body'].read()
        stats = json.loads(data.decode('utf-8'))
        logger.debug(f"Successfully retrieved processing stats for video ID: {video_id}")
        return stats
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Processing stats not found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Processing stats not found")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding processing stats JSON for video ID {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error decoding processing stats")
    except Exception as e:
        logger.error(f"Error retrieving processing stats for video ID {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processing stats")
########################################################

## Gets list of processed videos
########################################################
async def get_processed_videos():
    completed_videos_key = '_completed_videos.json'
    
    try:
        # Get the list of completed video IDs
        logger.debug("Retrieving list of completed video IDs")
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=completed_videos_key
            )
        data = await response['Body'].read()
        completed_video_ids = json.loads(data)
        logger.debug(f"Retrieved {len(completed_video_ids)} completed video IDs")

        processed_videos = []
        for video_id in completed_video_ids:
            try:
                # Get the processing_stats.json for this video
                logger.debug(f"Retrieving processing stats for video ID: {video_id}")
                stats_key = f'{video_id}/processing_stats.json'
                stats_response = await s3_client.get_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=stats_key
                )
                stats_data = await stats_response['Body'].read()
                stats_data = json.loads(stats_data)
                processed_videos.append({
                    'video_id': video_id,
                    'details': stats_data
                })
                logger.debug(f"Successfully retrieved details for video {video_id}")
            except Exception as e:
                logger.error(f"Error retrieving details for video {video_id}: {str(e)}", exc_info=True)

        logger.debug(f"Returning {len(processed_videos)} processed videos")
        return processed_videos

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error("No completed videos found")
            return []
        else:
            logger.error(f"Error retrieving completed videos list: {str(e)}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"Unexpected error in get_processed_videos: {str(e)}", exc_info=True)
        raise
########################################################