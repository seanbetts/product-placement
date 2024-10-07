import io
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import Counter
from wordcloud import WordCloud
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from utils.utils import find_font
from services import s3_operations
from models.detection_classes import DetectionResult, BrandDetector, Frame

## Detect brands in raw OCR data
########################################################
async def detect_brands(
    video_id: str,
    status_tracker: StatusTracker,
    video_details: VideoDetails,
    ocr_results: Optional[List[Dict]] = None
) -> List[DetectionResult]:
    try:
        if ocr_results is None:
            logger.info(f"Video Processing - Brand Detection - Step 3.1: Loading OCR results for video: {video_id}")
            ocr_results = await s3_operations.load_ocr_results(video_id)
            logger.info(f"Video Processing - Brand Detection - Step 3.1: Loaded {len(ocr_results)} OCR results for video: {video_id}")

        logger.debug(f"Video Processing - Brand Detection - Step 3.1: OCR results type: {type(ocr_results)}")
        logger.debug(f"Video Processing - Brand Detection - Step 3.1: Sample OCR result: {ocr_results[0] if ocr_results else 'No OCR results'}")

        video_width, video_height = await video_details.get_detail('video_resolution')
        video_fps = await video_details.get_detail('frames_per_second')

        brand_detector = BrandDetector(fps=video_fps)
        frames = [Frame(ocr['frame_number'], video_width, video_height, ocr) for ocr in ocr_results]
        
        results = []
        total_frames = len(frames)
        total_detected_brands = 0

        # Step 3.2: Process frames
        logger.info(f"Video Processing - Brand Detection - Step 3.2: Starting to process {total_frames} frames for video: {video_id}")
        for i, frame in enumerate(frames):
            result = await brand_detector.process_frame(frame)
            results.append(result)
            total_detected_brands += len(result.detected_brands)
            progress = int((i + 1) / total_frames * 100)
            await status_tracker.update_process_status("ocr", "in_progress", progress)
            
            if i > 0 and i % 100 == 0:
                logger.info(f"Video Processing - Brand Detection - Step 3.2: Processed {i}/{total_frames} frames for video: {video_id}. Total brands detected so far: {total_detected_brands}")

        # Step 3.3: Create word cloud
        logger.info(f"Video Processing - Brand Detection - Step 3.3: Creating word cloud for video: {video_id}")
        await create_word_cloud(video_id, results)

        # Step 3.4: Save filtered brands OCR results
        logger.info(f"Video Processing - Brand Detection - Step 3.4: Saving filtered brand OCR results for video: {video_id}")
        await s3_operations.save_brands_ocr_results(video_id, results)

        # Step 3.5: Create and save brand table
        logger.info(f"Video Processing - Brand Detection - Step 3.5: Creating and saving brand table for video: {video_id}")
        brand_appearances = {}
        for result in results:
            for detected_brand in result.detected_brands:
                if detected_brand.text not in brand_appearances:
                    brand_appearances[detected_brand.text] = {}
                brand_appearances[detected_brand.text][result.frame_number] = [detected_brand]
        
        brand_stats = await s3_operations.create_and_save_brand_table(video_id, results, video_fps)
        logger.info(f"Video Processing - Brand Detection - Step 3.6: Created brand table with {len(brand_stats)} entries for video: {video_id}")

        await status_tracker.update_process_status("ocr", "completed", 100)
        
        # Log some statistics about the results
        total_brands = sum(len(result.detected_brands) for result in results)
        unique_brands = set(brand.text for result in results for brand in result.detected_brands)
        logger.info(f"Video Processing - Brand Detection - Step 3.7: Detected {len(unique_brands)} unique brands across {total_brands} brand instances in video: {video_id}")
        logger.info(f"Video Processing - Brand Detection - Step 3.8: Completed brand detection for video: {video_id}")

        return results

    except Exception as e:
        logger.error(f"Video Processing - Brand Detection: Error in detect_brands for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.update_process_status("ocr", "error", 0)
        raise
########################################################

## Create wordcloud
########################################################
async def create_word_cloud(video_id: str, results: List[DetectionResult]):
    """
    Create a styled word cloud from the processed OCR results, using individual text annotations
    and a default system font. Only includes words with confidence above the minimum threshold
    and length greater than 2 characters.
    """
    
    # Extract all text annotations with confidence above the threshold and length > 2
    all_text = []
    excluded_confidence_count = 0
    excluded_length_count = 0
    excluded_not_word_count = 0
    brand_match_count = 0

    logger.debug(f"Video Processing - Brand Detection - Step 3.3 - OCR Data Processing: Processing cleaned OCR results for word cloud creation for video: {video_id}")
    for frame in results:
        for detection in frame.detected_brands:
            confidence = detection.confidence
            
            # Use the text field of BrandInstance, which should already be the best match
            text = detection.text.lower().strip()
            
            if detection.is_interpolated:
                brand_match_count += 1

            if confidence >= settings.WORDCLOUD_MINIMUM_CONFIDENCE:
                if len(text) > 2:
                    all_text.append(text)
                else:
                    excluded_length_count += 1
            else:
                excluded_confidence_count += 1
    
    if not all_text:
        logger.warning(f"Video Processing - Brand Detection - Step 3.3: No text found for word cloud creation for video: {video_id}")
        return
    
    logger.debug(f"Video Processing - Brand Detection - Step 3.3: Excluded {excluded_confidence_count} words due to low confidence, "
                        f"{excluded_length_count} words due to short length, and "
                        f"{excluded_not_word_count} words not in dictionary. "
                        f"Included {brand_match_count} brand matches for video: {video_id}")

    # Count word frequencies
    word_freq = Counter(all_text)
    
    # Create a mask image (circle shape)
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # Find the first available preferred font
    font_path = await find_font(settings.PREFERRED_FONTS)
    if not font_path:
        logger.warning("Video Processing - Brand Detection - Step 3.3: No preferred font found. Using default.")

    try:
        logger.debug(f"Video Processing - Brand Detection - Step 3.3: Generating word cloud for video: {video_id}")
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

        logger.debug(f"Video Processing - Brand Detection - Step 3.3: Uploading word cloud to S3 for video: {video_id}")
        
        # Upload to S3
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/ocr/wordcloud.jpg',
                Body=img_buffer.getvalue(),
                ContentType='image/jpeg'
            )
        logger.debug(f"Video Processing - Brand Detection - Step 3.3: Word cloud created and saved for video: {video_id}")
    except Exception as e:
        logger.error(f"Video Processing - Brand Detection - Step 3.3: Error creating or saving word cloud for video {video_id}: {str(e)}", exc_info=True)
        raise
    finally:
        plt.close()
########################################################