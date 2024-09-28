import json
import io
import time
import asyncio
import enchant
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from typing import AsyncGenerator
from matplotlib import font_manager
from typing import Tuple, Dict, List, AsyncGenerator
from PIL import Image
from collections import Counter
from thefuzz import fuzz
from wordcloud import WordCloud
from core.config import settings
from core.aws import get_s3_client
from core.logging import AppLogger, dual_log

# Create a global instance of AppLogger
app_logger = AppLogger()

## Gets the resolution of a video
########################################################
async def get_video_resolution(vlogger, video_id: str) -> Tuple[int, int]:
    @vlogger.log_performance
    async def _get_video_resolution():
        try:
            # Construct the path to the first frame
            first_frame_path = f'{video_id}/frames/000000.jpg'
            vlogger.logger.info(f"Attempting to retrieve first frame for video: {video_id}")
            
            # Download the frame data
            async with get_s3_client() as s3_client:
                response = await s3_client.get_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=first_frame_path
                )
            frame_data = await response['Body'].read()
            await vlogger.log_s3_operation("download", len(frame_data))
            
            # Open the image using PIL
            with Image.open(io.BytesIO(frame_data)) as img:
                width, height = img.size
            
            vlogger.logger.info(f"Detected resolution for video {video_id}: {width}x{height}")
            return (width, height)
        
        except s3_client.exceptions.NoSuchKey:
            vlogger.logger.error(f"First frame not found for video {video_id}")
            raise FileNotFoundError(f"First frame not found for video {video_id}")
        
        except Exception as e:
            vlogger.logger.error(f"Unexpected error retrieving video resolution for {video_id}: {str(e)}", exc_info=True)
            # Return a default resolution if unable to retrieve
            vlogger.logger.warning(f"Using default resolution (1920x1080) for video {video_id}")
            return (1920, 1080)

    return await _get_video_resolution()
########################################################

## Updates the name of a video
########################################################
async def update_video_name(video_id: str, new_name: str):
    app_logger.log_info(f"Attempting to update name for video {video_id} to '{new_name}'")

    try:
        status_key = f'{video_id}/status.json'
        app_logger.log_info(f"Updating status.json for video {video_id}")
        
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
        app_logger.log_info(f"Updating processing_stats.json for video {video_id}")
        
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

        app_logger.log_info(f"Successfully updated name for video {video_id} to '{new_name}'")
        return True

    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"Video {video_id} not found")
        raise FileNotFoundError(f"Video {video_id} not found")
    
    except Exception as e:
        app_logger.log_error(f"Error updating name for video {video_id}: {str(e)}", exc_info=True)
        raise
########################################################

## Asynchronously paginate through S3 objects
########################################################
async def async_paginate(vlogger, paginator, **kwargs) -> AsyncGenerator:
    """
    Asynchronously paginate through S3 objects.
    Args:
    vlogger: The VideoLogger instance for logging.
    paginator: The S3 paginator object.
    **kwargs: Arguments to pass to paginator.paginate.
    Yields:
    Each page of the paginator.
    """
    @vlogger.log_performance
    async def _async_paginate():
        try:
            vlogger.logger.info("Starting asynchronous pagination")
            page_count = 0
            total_items = 0
            # Retrieve the paginator as a generator in a separate thread
            paginate_gen = await asyncio.to_thread(paginator.paginate, **kwargs)
            # Iterate over the paginator generator
            for page in paginate_gen:
                page_count += 1
                items_in_page = len(page.get('Contents', []))
                total_items += items_in_page
                vlogger.logger.debug(f"Retrieved page {page_count} with {items_in_page} items")
                yield page
            vlogger.logger.info(f"Pagination completed. Total pages: {page_count}, Total items: {total_items}")
        except Exception as e:
            vlogger.logger.error(f"Error during pagination: {str(e)}", exc_info=True)
            raise

    async for page in _async_paginate():
        yield page

########################################################

## Convert relative bounding box to absolute bounding box
########################################################
async def convert_relative_bbox(vlogger, bbox: Dict, video_resolution: Tuple[int, int]) -> Dict:
    """
    Convert relative bounding box coordinates to absolute pixel values.
    Args:
    vlogger: The VideoLogger instance for logging.
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
        vlogger.logger.debug(f"Converted bbox {bbox} to vertices {vertices} for resolution {video_resolution}")
        return {"vertices": vertices}
    except Exception as e:
        vlogger.logger.error(f"Error converting bounding box {bbox} for resolution {video_resolution}: {str(e)}", exc_info=True)
        # Return an empty bounding box in case of error
        return {"vertices": []}
########################################################

## Create wordcloud
########################################################
async def create_word_cloud(vlogger, video_id: str, cleaned_results: List[Dict]):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font. Only includes words with confidence above the minimum threshold
    and length greater than 2 characters.
    """
    @vlogger.log_performance
    async def _create_word_cloud():
        # Create an enchant dictionary for English
        d = enchant.Dict("en_US")
        
        # Extract all text annotations with confidence above the threshold and length > 2
        all_text = []
        excluded_confidence_count = 0
        excluded_length_count = 0
        excluded_not_word_count = 0
        brand_match_count = 0

        vlogger.logger.debug(f"OCR Data Processing: Processing cleaned OCR results for word cloud creation for video: {video_id}")
        for frame in cleaned_results:
            for detection in frame['cleaned_detections']:
                confidence = detection.get('confidence', 1.0)  # Default to 1.0 if confidence is not available
                
                # Prioritize brand_match, then cleaned_text (if it's a valid word), then original text
                if detection.get('brand_match') is not None:
                    text = detection['brand_match'].lower().strip()
                    brand_match_count += 1
                elif 'cleaned_text' in detection and d.check(detection['cleaned_text'].strip()):
                    text = detection['cleaned_text'].lower().strip()
                elif 'cleaned_text' in detection:
                    excluded_not_word_count += 1
                    continue
                else:
                    text = detection.get('text', '').lower().strip()

                if confidence >= settings.WORDCLOUD_MINIMUM_CONFIDENCE:
                    if len(text) > 2:
                        all_text.append(text)
                    else:
                        excluded_length_count += 1
                else:
                    excluded_confidence_count += 1
        
        if not all_text:
            vlogger.logger.warning(f"No text found for word cloud creation for video: {video_id}")
            return
        
        # vlogger.logger.debug(f"Excluded {excluded_confidence_count} words due to low confidence, "
        #                     f"{excluded_length_count} words due to short length, and "
        #                     f"{excluded_not_word_count} words not in dictionary. "
        #                     f"Included {brand_match_count} brand matches for video: {video_id}")

        # Count word frequencies
        word_freq = Counter(all_text)
        
        # Create a mask image (circle shape)
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * mask.astype(int)

        # Define font preference order
        preferred_fonts = [
            'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans',
            'Roboto', 'Open Sans', 'Lato', 'Noto Sans'
        ]

        # Find the first available preferred font
        font_path = await find_font(preferred_fonts)
        if not font_path:
            vlogger.logger.warning("No preferred font found. Using default.")

        try:
            # vlogger.logger.debug(f"Generating word cloud for video: {video_id}")
            # Generate word cloud
            wordcloud = WordCloud(width=600, height=600,
                                  background_color='white',
                                  max_words=100,  # Limit to top 100 words for clarity
                                  min_font_size=10,
                                  max_font_size=120,
                                  mask=mask,
                                  font_path=font_path,
                                  colormap='ocean',  # Use a colorful colormap
                                  prefer_horizontal=0.9,
                                  scale=2
                                  ).generate_from_frequencies(word_freq)

            # Create a figure and save it to a BytesIO object
            plt.figure(figsize=(10,10), frameon=False)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
            img_buffer.seek(0)

            # vlogger.logger.debug(f"Uploading word cloud to S3 for video: {video_id}")
            
            # Upload to S3
            async with get_s3_client() as s3_client:
                await vlogger.log_performance(s3_client.put_object)(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=f'{video_id}/ocr/wordcloud.jpg',
                    Body=img_buffer.getvalue(),
                    ContentType='image/jpeg'
                )
            await vlogger.log_s3_operation("upload", img_buffer.getbuffer().nbytes)
            # vlogger.logger.debug(f"Word cloud created and saved for video: {video_id}")
        except Exception as e:
            dual_log(vlogger, app_logger, 'info', f"OCR Data Processing: Error creating or saving word cloud for video {video_id}: {str(e)}", exc_info=True)
            raise
        finally:
            plt.close()

    return await _create_word_cloud()
########################################################

## Find font for wordcloud
########################################################
async def find_font(font_names: List[str]) -> str:
    """
    Find the first available font from the given list of font names.
    Args:
    vlogger: The VideoLogger instance for logging.
    font_names (list): List of font names to search for.
    Returns:
    str or None: Path to the first available font, or None if no fonts are found.
    """
    # app_logger.log_info(f"Searching for fonts from list: {font_names}")
    for font_name in font_names:
        try:
            # Use asyncio.to_thread for potentially blocking operations
            font_path = await asyncio.to_thread(font_manager.findfont, font_manager.FontProperties(family=font_name))
            # vlogger.logger.debug(f"Found font: {font_name} at path: {font_path}")
            return font_path
        except Exception as e:
            # vlogger.logger.debug(f"Font not found: {font_name}. Error: {str(e)}")
            continue
    app_logger.log_warning("No fonts found from the provided list.")
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
        # app_logger.log_info(f"Calculating custom ratio for strings: '{s1}' and '{s2}'")
        base_ratio = fuzz.ratio(s1, s2)
        # app_logger.log_info(f"Base ratio: {base_ratio}")
        if len(s1) >= 3 and len(s2) >= 3:
            start_ratio = fuzz.ratio(s1[:3], s2[:3])
            # app_logger.log_info(f"Start ratio: {start_ratio}")
            final_ratio = (base_ratio + start_ratio) / 2
            # app_logger.log_info(f"Final weighted ratio: {final_ratio}")
            return final_ratio
        else:
            # app_logger.log_info(f"Strings too short for start comparison. Using base ratio: {base_ratio}")
            return base_ratio
    except Exception as e:
        app_logger.log_error(f"Error in custom_ratio function: {str(e)}", exc_info=True)
        # In case of error, return 0 as the similarity ratio
        return 0
########################################################
