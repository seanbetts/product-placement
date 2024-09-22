import json
import datetime
import asyncio
from typing import Dict, Any
from fastapi import HTTPException
from core.config import settings
from core.logging import AppLogger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from services import video_processing
from botocore.exceptions import ClientError

# Create a global instance of AppLogger
app_logger = AppLogger()

# Create a global instance of s3_client
s3_client = get_s3_client()

## Get processing status
########################################################
async def get_processing_status(vlogger, video_id: str) -> Dict[str, Any]:
    @vlogger.log_performance
    async def _get_processing_status():
        vlogger.logger.info(f"Received status request for video ID: {video_id}")
        status_key = f'{video_id}/status.json'
        
        try:
            vlogger.logger.info(f"Attempting to retrieve status for video ID: {video_id}")
            response = await vlogger.log_performance(asyncio.to_thread)(
                s3_client.get_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=status_key
            )
            data = response['Body'].read()
            vlogger.log_s3_operation("download", len(data))
            status_data = json.loads(data.decode('utf-8'))
            vlogger.logger.info(f"Status retrieved for video ID {video_id}: {status_data}")
            return status_data
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                vlogger.logger.warning(f"Status not found for video ID: {video_id}")
                raise HTTPException(status_code=404, detail="Video status not found")
            else:
                vlogger.logger.error(f"Error retrieving status for video ID {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error retrieving video status")
        except json.JSONDecodeError as e:
            vlogger.logger.error(f"Error decoding status JSON for video ID {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error decoding video status")
        except Exception as e:
            vlogger.logger.error(f"Unexpected error retrieving status for video ID {video_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected error retrieving video status")

    return await _get_processing_status()
########################################################

## Update processing status
########################################################
async def periodic_status_update(vlogger, video_id: str, status_tracker: StatusTracker, s3_client):
    @vlogger.log_performance
    async def _update_status():
        try:
            status_tracker.calculate_overall_progress()
            # Use the status_tracker's method directly, which is now async
            await status_tracker.update_s3_status()
            vlogger.log_s3_operation("upload", len(json.dumps(status_tracker.status)))
            vlogger.logger.info(f"Updated status for video {video_id}: {status_tracker.status}")
        except Exception as e:
            vlogger.logger.error(f"Error updating status for video {video_id}: {str(e)}", exc_info=True)

    vlogger.logger.info(f"Starting periodic status updates for video {video_id}")
    update_count = 0
    try:
        while True:
            await _update_status()
            update_count += 1
            vlogger.logger.debug(f"Completed status update {update_count} for video {video_id}")
            await asyncio.sleep(settings.STATUS_UPDATE_INTERVAL)
    except asyncio.CancelledError:
        vlogger.logger.info(f"Periodic status updates stopped for video {video_id} after {update_count} updates")
    except Exception as e:
        vlogger.logger.error(f"Unexpected error in periodic status updates for video {video_id}: {str(e)}", exc_info=True)
        # Consider updating the status_tracker with this error
        await status_tracker.set_error(f"Status update error: {str(e)}")
########################################################

## Mark video as completed processing
########################################################
async def mark_video_as_completed(vlogger, video_id: str):
    @vlogger.log_performance
    async def _mark_video_as_completed():
        stats_key = f'{video_id}/processing_stats.json'
        
        try:
            # Get the current processing stats
            vlogger.logger.info(f"Retrieving processing stats for video {video_id}")
            response = await vlogger.log_performance(asyncio.to_thread)(
                s3_client.get_object,
                Bucket=settings.PROCESSING_BUCKET,
                Key=stats_key
            )
            data = response['Body'].read()
            vlogger.log_s3_operation("download", len(data))
            stats_data = json.loads(data.decode('utf-8'))

            # Update the stats
            stats_data['status'] = 'completed'
            stats_data['completion_time'] = datetime.datetime.utcnow().isoformat()

            # Write the updated stats back to S3
            vlogger.logger.info(f"Updating processing stats for video {video_id}")
            updated_stats_json = json.dumps(stats_data)
            await vlogger.log_performance(s3_client.put_object)(
                Bucket=settings.PROCESSING_BUCKET,
                Key=stats_key,
                Body=updated_stats_json,
                ContentType='application/json'
            )
            vlogger.log_s3_operation("upload", len(updated_stats_json))

            # Update the completed videos list
            vlogger.logger.info(f"Updating completed videos list for video {video_id}")
            await video_processing.update_completed_videos_list(vlogger, video_id)

            vlogger.logger.info(f"Marked video {video_id} as completed")

        except Exception as e:
            vlogger.logger.error(f"Error marking video {video_id} as completed: {str(e)}", exc_info=True)
            # Optionally, you might want to re-raise the exception here, depending on how you want to handle errors

    await _mark_video_as_completed()
########################################################

## Gets processing stats for a video
########################################################
async def get_processing_stats(video_id: str):
    # app_logger.log_info(f"Received request for processing stats of video: {video_id}")
    stats_key = f'{video_id}/processing_stats.json'
    
    try:
        # app_logger.log_info(f"Attempting to retrieve processing stats for video ID: {video_id}")
        response = await asyncio.to_thread (
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET, 
            Key=stats_key
        )
        data = response['Body'].read()
        stats = json.loads(data.decode('utf-8'))
        # app_logger.log_info(f"Successfully retrieved processing stats for video ID: {video_id}")
        return stats
    except s3_client.exceptions.NoSuchKey:
        app_logger.error(f"Processing stats not found for video ID: {video_id}")
        raise HTTPException(status_code=404, detail="Processing stats not found")
    except json.JSONDecodeError as e:
        app_logger.log_error(f"Error decoding processing stats JSON for video ID {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error decoding processing stats")
    except Exception as e:
        app_logger.log_error(f"Error retrieving processing stats for video ID {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving processing stats")
########################################################

## Gets list of processed videos
########################################################
async def get_processed_videos():
    completed_videos_key = '_completed_videos.json'
    
    try:
        # Get the list of completed video IDs
        app_logger.log_info("Retrieving list of completed video IDs")
        response = await asyncio.to_thread(
            s3_client.get_object,
            Bucket=settings.PROCESSING_BUCKET,
            Key=completed_videos_key
        )
        data = await asyncio.to_thread(response['Body'].read)
        completed_video_ids = json.loads(data)
        # app_logger.log_info(f"Retrieved {len(completed_video_ids)} completed video IDs")

        processed_videos = []
        for video_id in completed_video_ids:
            try:
                # Get the processing_stats.json for this video
                # app_logger.log_info(f"Retrieving processing stats for video ID: {video_id}")
                stats_key = f'{video_id}/processing_stats.json'
                stats_response = await asyncio.to_thread(
                    s3_client.get_object,
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=stats_key
                )
                stats_data = await asyncio.to_thread(stats_response['Body'].read)
                stats_data = json.loads(stats_data)
                processed_videos.append({
                    'video_id': video_id,
                    'details': stats_data
                })
                # app_logger.log_info(f"Successfully retrieved details for video {video_id}")
            except Exception as e:
                app_logger.log_error(f"Error retrieving details for video {video_id}: {str(e)}", exc_info=True)

        app_logger.log_info(f"Returning {len(processed_videos)} processed videos")
        return processed_videos

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            app_logger.log_error("No completed videos found")
            return []
        else:
            app_logger.log_error(f"Error retrieving completed videos list: {str(e)}", exc_info=True)
            raise
    except Exception as e:
        app_logger.log_error(f"Unexpected error in get_processed_videos: {str(e)}", exc_info=True)
        raise
########################################################