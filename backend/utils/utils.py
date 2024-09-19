import json
import io
import asyncio
from typing import Tuple, Dict, List
from PIL import Image
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from thefuzz import fuzz
from wordcloud import WordCloud
from core.logging import logger
from core.config import settings
from core.aws import get_s3_client
from botocore.exceptions import ClientError

## Gets the resolution of a video
########################################################
async def get_video_resolution(video_id: str) -> Tuple[int, int]:
    s3_client = get_s3_client()

    try:
        # Construct the path to the first frame
        first_frame_path = f'{video_id}/frames/000000.jpg'
        
        # Download the frame data
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=first_frame_path)
        frame_data = response['Body'].read()
        
        # Open the image using PIL
        with Image.open(io.BytesIO(frame_data)) as img:
            width, height = img.size
        
        # logger.info(f"Detected resolution for video {video_id}: {width}x{height}")
        return (width, height)
    
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"First frame not found for video {video_id}")
        raise FileNotFoundError(f"First frame not found for video {video_id}")
    except Exception as e:
        logger.error(f"Unexpected error retrieving video resolution for {video_id}: {str(e)}")
        # Return a default resolution if unable to retrieve
        logger.warning(f"Using default resolution (1920x1080) for video {video_id}")
        return (1920, 1080)  # Default to 1080p
########################################################

## Updates the name of a video
########################################################
async def update_video_name(video_id: str, new_name: str):
    s3_client = get_s3_client()
    
    # Update status.json
    status_key = f'{video_id}/status.json'
    try:
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=status_key)
        status_data = json.loads(response['Body'].read().decode('utf-8'))
        status_data['name'] = new_name
        s3_client.put_object(Bucket=settings.PROCESSING_BUCKET, Key=status_key, 
                             Body=json.dumps(status_data), ContentType='application/json')

        # Update processing_stats.json
        stats_key = f'{video_id}/processing_stats.json'
        response = s3_client.get_object(Bucket=settings.PROCESSING_BUCKET, Key=stats_key)
        stats_data = json.loads(response['Body'].read().decode('utf-8'))
        stats_data['name'] = new_name
        s3_client.put_object(Bucket=settings.PROCESSING_BUCKET, Key=stats_key, 
                             Body=json.dumps(stats_data), ContentType='application/json')

        return True
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"Video {video_id} not found")
    except Exception as e:
        logger.error(f"Error updating name for video {video_id}: {str(e)}")
        raise
########################################################

## Asynchronously paginate through S3 objects
########################################################
async def async_paginate(paginator, **kwargs):
    """
    Asynchronously paginate through S3 objects.

    Args:
        paginator: The S3 paginator object.
        **kwargs: Arguments to pass to paginator.paginate.

    Yields:
        Each page of the paginator.
    """
    try:
        # Retrieve the paginator as a generator in a separate thread
        paginate_gen = await asyncio.to_thread(paginator.paginate, **kwargs)
        
        # Iterate over the paginator generator
        for page in paginate_gen:
            yield page
    except Exception as e:
        logger.error(f"Error during pagination: {str(e)}")
        raise
########################################################


## Convert relative bounding box to absolute bounding box
########################################################
def convert_relative_bbox(bbox: Dict, video_resolution: Tuple[int, int]) -> Dict:
    """
    Convert relative bounding box coordinates to absolute pixel values.

    Args:
        bbox (Dict): Bounding box with relative 'Width', 'Height', 'Left', 'Top'.
        video_resolution (Tuple[int, int]): (width, height) of the video.

    Returns:
        Dict: Bounding box with absolute 'vertices' coordinates.
    """
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

    return {"vertices": vertices}
########################################################

## Create wordcloud
########################################################
def create_word_cloud(s3_client, video_id: str, cleaned_results: List[Dict]):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font.
    """
    # Use the 'Agg' backend which doesn't require a GUI
    matplotlib.use('Agg')
    
    # Extract all text annotations
    all_text = []
    for frame in cleaned_results:
        for detection in frame['cleaned_detections']:
            # Use cleaned_text if available, otherwise fall back to text
            text = detection.get('cleaned_text', detection.get('text', '')).lower()
            if text:
                all_text.append(text)
    
    if not all_text:
        logger.warning(f"No text found for word cloud creation for video: {video_id}")
        return

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
    font_path = find_font(preferred_fonts)
    if not font_path:
        logger.warning("No preferred font found. Using default.")

    try:
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

        # Upload to S3
        s3_client.put_object(
            Bucket=settings.PROCESSING_BUCKET,
            Key=f'{video_id}/ocr/wordcloud.jpg',
            Body=img_buffer.getvalue(),
            ContentType='image/jpeg'
        )
        logger.info(f"Word cloud created and saved for video: {video_id}")
    except Exception as e:
        logger.error(f"Error creating or saving word cloud for video {video_id}: {str(e)}")
        raise
    finally:
        plt.close()
########################################################

## Find font for wordcloud
########################################################
def find_font(font_names):
    """
    Find the first available font from the given list of font names.
    """
    for font_name in font_names:
        try:
            return font_manager.findfont(font_manager.FontProperties(family=font_name))
        except:
            continue
    return None
########################################################


## Weight the beginning of a string
########################################################
def custom_ratio(s1, s2):
    """Custom ratio function that gives more weight to the beginning of the string."""
    base_ratio = fuzz.ratio(s1, s2)
    if len(s1) >= 3 and len(s2) >= 3:
        start_ratio = fuzz.ratio(s1[:3], s2[:3])
        return (base_ratio + start_ratio) / 2
    return base_ratio
########################################################
