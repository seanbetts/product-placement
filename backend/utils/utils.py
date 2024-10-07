import json
import io
import asyncio
from typing import AsyncGenerator
from matplotlib import font_manager
from typing import Tuple, Dict, List, AsyncGenerator
from PIL import Image
from thefuzz import fuzz
from core.config import settings
from core.aws import get_s3_client
from core.logging import logger

## Gets the resolution of a video
########################################################
async def get_video_resolution(video_id: str) -> Tuple[int, int]:
    try:
        # Construct the path to the first frame
        first_frame_path = f'{video_id}/frames/000000.jpg'
        logger.debug(f"Attempting to retrieve first frame for video: {video_id}")
        
        # Download the frame data
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=first_frame_path
            )
        frame_data = await response['Body'].read()
        
        # Open the image using PIL
        with Image.open(io.BytesIO(frame_data)) as img:
            width, height = img.size
        
        logger.debug(f"Detected resolution for video {video_id}: {width}x{height}")
        return (width, height)
    
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"First frame not found for video {video_id}")
        raise FileNotFoundError(f"First frame not found for video {video_id}")
    
    except Exception as e:
        logger.error(f"Unexpected error retrieving video resolution for {video_id}: {str(e)}", exc_info=True)
        # Return a default resolution if unable to retrieve
        logger.error(f"Using default resolution (1920x1080) for video {video_id}")
        return (1920, 1080)
########################################################

## Updates the name of a video
########################################################
async def update_video_name(video_id: str, new_name: str):
    logger.info(f"Attempting to update name for video {video_id} to '{new_name}'")

    try:
        status_key = f'{video_id}/status.json'
        logger.info(f"Updating status.json for video {video_id}")
        
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=status_key
            )
        status_data = await response['Body'].read()
        
        status_data = json.loads(status_data.decode('utf-8'))
        status_data['name'] = new_name
        
        updated_status_data = json.dumps(status_data)
        await s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET, 
            Key=status_key,
            Body=updated_status_data, 
            ContentType='application/json'
        )
        
        # Update processing_stats.json
        stats_key = f'{video_id}/processing_stats.json'
        logger.info(f"Updating processing_stats.json for video {video_id}")
        
        response = await s3_client.get_object(
            Bucket=settings.PROCESSING_BUCKET, 
            Key=stats_key
        )
        stats_data = await response['Body'].read()
        
        stats_data = json.loads(stats_data.decode('utf-8'))
        stats_data['name'] = new_name
        
        updated_stats_data = json.dumps(stats_data)
        await s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET, 
            Key=stats_key,
            Body=updated_stats_data, 
            ContentType='application/json'
        )

        logger.info(f"Successfully updated name for video {video_id} to '{new_name}'")
        return True

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Video {video_id} not found")
        raise FileNotFoundError(f"Video {video_id} not found")
    
    except Exception as e:
        logger.error(f"Error updating name for video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## Asynchronously paginate through S3 objects
########################################################
async def async_paginate(paginator, **kwargs) -> AsyncGenerator:
    """
    Asynchronously paginate through S3 objects.
    Args:
    paginator: The S3 paginator object.
    **kwargs: Arguments to pass to paginator.paginate.
    Yields:
    Each page of the paginator.
    """
    async def _async_paginate():
        try:
            logger.debug("Starting asynchronous pagination")
            page_count = 0
            total_items = 0
            # Retrieve the paginator as a generator in a separate thread
            paginate_gen = await asyncio.to_thread(paginator.paginate, **kwargs)
            # Iterate over the paginator generator
            for page in paginate_gen:
                page_count += 1
                items_in_page = len(page.get('Contents', []))
                total_items += items_in_page
                logger.debug(f"Retrieved page {page_count} with {items_in_page} items")
                yield page
            logger.debug(f"Pagination completed. Total pages: {page_count}, Total items: {total_items}")
        except Exception as e:
            logger.error(f"Error during pagination: {str(e)}", exc_info=True)
            raise

    async for page in _async_paginate():
        yield page

########################################################

## Convert relative bounding box to absolute bounding box
########################################################
async def convert_relative_bbox(bbox: Dict, video_resolution: Tuple[int, int]) -> Dict:
    """
    Convert relative bounding box coordinates to absolute pixel values.
    Args:
    bbox (Dict): Bounding box with relative 'Width', 'Height', 'Left', 'Top'.
    video_resolution (Tuple[int, int]): (width, height) of the video.
    Returns:
    Dict: Bounding box with absolute 'vertices' coordinates.
    """
    try:
        video_width, video_height = video_resolution
        left = bbox.get('Left', 0) * video_width
        top = bbox.get('Top', 0) * video_height
        width = bbox.get('Width', 0) * video_width
        height = bbox.get('Height', 0) * video_height
        vertices = [
            {"x": int(left), "y": int(top)},
            {"x": int(left + width), "y": int(top)},
            {"x": int(left + width), "y": int(top + height)},
            {"x": int(left), "y": int(top + height)}
        ]
        logger.debug(f"Converted bbox {bbox} to vertices {vertices} for resolution {video_resolution}")
        return {"vertices": vertices}
    except Exception as e:
        logger.error(f"Error converting bounding box {bbox} for resolution {video_resolution}: {str(e)}", exc_info=True)
        # Return an empty bounding box in case of error
        return {"vertices": []}
########################################################

## Find fonts
########################################################
async def find_font(font_names: List[str]) -> str:
    """
    Find the first available font from the given list of font names.
    Args:
    font_names (list): List of font names to search for.
    Returns:
    str or None: Path to the first available font, or None if no fonts are found.
    """
    logger.debug(f"Searching for fonts from list: {font_names}")
    for font_name in font_names:
        try:
            # Use asyncio.to_thread for potentially blocking operations
            font_path = await asyncio.to_thread(font_manager.findfont, font_manager.FontProperties(family=font_name))
            logger.debug(f"Found font: {font_name} at path: {font_path}")
            return font_path
        except Exception as e:
            logger.debug(f"Font not found: {font_name}. Error: {str(e)}")
            continue
    logger.warning("No fonts found from the provided list.")
    return None
########################################################

## Weight the beginning of a string
########################################################
def custom_ratio(s1, s2):
    """
    Custom ratio function that gives more weight to the beginning of the string.
    Args:
    s1 (str): First string to compare.
    s2 (str): Second string to compare.
    Returns:
    float: The custom similarity ratio between the two strings.
    """
    try:
        logger.debug(f"Calculating custom ratio for strings: '{s1}' and '{s2}'")
        base_ratio = fuzz.ratio(s1, s2)
        logger.debug(f"Base ratio: {base_ratio}")
        if len(s1) >= 3 and len(s2) >= 3:
            start_ratio = fuzz.ratio(s1[:3], s2[:3])
            logger.debug(f"Start ratio: {start_ratio}")
            final_ratio = (base_ratio + start_ratio) / 2
            logger.debug(f"Final weighted ratio: {final_ratio}")
            return final_ratio
        else:
            logger.debug(f"Strings too short for start comparison. Using base ratio: {base_ratio}")
            return base_ratio
    except Exception as e:
        logger.error(f"Error in custom_ratio function: {str(e)}", exc_info=True)
        # In case of error, return 0 as the similarity ratio
        return 0
########################################################
