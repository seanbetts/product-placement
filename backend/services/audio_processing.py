import time
import json
import asyncio
import aioboto3
from fastapi import HTTPException
from core.config import settings
from core.logging import logger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
from models.video_details import VideoDetails
from botocore.exceptions import ClientError

## Transcribes the video audio
########################################################
import asyncio
import time
import boto3
from botocore.exceptions import ClientError

async def transcribe_audio(video_id: str, status_tracker: StatusTracker, video_details: VideoDetails):
    logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 2.2: Transcribing audio for video: {video_id}")
    audio_key = f'{video_id}/audio.mp3'
    transcript_key = f"{video_id}/transcripts/audio_transcript_{video_id}.json"
    video_length = await video_details.get_detail("video_length")

    try:
        # Check if the audio file exists
        async with aioboto3.client('s3') as s3_client:
            try:
                await s3_client.head_object(
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=audio_key
                )
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning(f"Video Processing - Thread 2 - Audio Processing - Step 2.2: Audio file not found for video: {video_id}")
                    await status_tracker.update_process_status("transcription", "error", 0)
                    await status_tracker.set_error("Video Processing - Thread 2 - Audio Processing - Step 2.2: Audio file not found for transcription.")
                    return None
                else:
                    raise

        # Instantiate the asynchronous Transcribe client
        async with aioboto3.client('transcribe') as transcribe_client:
            # Set up the transcription job
            job_name = f"audio_transcript_{video_id}"
            job_uri = f"s3://{settings.PROCESSING_BUCKET}/{audio_key}"
            
            transcription_start_time = time.time()
            logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 2.3: Starting transcription job for video: {video_id}")
            await transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': job_uri},
                MediaFormat='mp3',
                LanguageCode='en-US',
                OutputBucketName=settings.PROCESSING_BUCKET,
                OutputKey=f"{video_id}/transcripts/",
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 10,
                    'ShowAlternatives': False
                }
            )

            # Wait for the job to complete with a timeout of 2x video length
            timeout = video_length * 2
            while True:
                response = await transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                job_status = response['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                
                await asyncio.sleep(5)
                elapsed_time = time.time() - transcription_start_time
                progress = min(95, (elapsed_time / timeout) * 100)  # Cap at 95% until completion
                await status_tracker.update_process_status("transcription", "in_progress", progress)
                logger.debug(f"Video Processing - Thread 2 - Audio Processing - Step 2.3: Transcription progress for video {video_id}: {progress:.2f}%")

            transcription_end_time = time.time()
            
            if job_status == 'COMPLETED':
                logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 2.4: Transcription completed for video {video_id}")
                
                # Wait for the transcript file to be available in S3
                max_retries = 10
                async with aioboto3.client('s3') as s3_client:
                    for i in range(max_retries):
                        try:
                            await s3_client.head_object(
                                Bucket=settings.PROCESSING_BUCKET,
                                Key=transcript_key
                            )
                            logger.debug(f"Video Processing - Thread 2 - Audio Processing - Step 2.4: Transcript file found for video {video_id}")
                            break
                        except ClientError:
                            if i < max_retries - 1:
                                await asyncio.sleep(5)
                            else:
                                raise FileNotFoundError(f"Video Processing - Thread 2 - Audio Processing - Step 2.4: Transcript file not found for video {video_id} after {max_retries} retries")

                # Process the response and create transcripts
                plain_transcript, json_transcript, word_count, overall_confidence = await process_transcription_response(video_id)

                logger.info(f"Video Processing - Thread 2 - Audio Processing - Step 2.5: Transcripts processed and uploaded for video: {video_id}")

                # Calculate transcription stats
                transcription_time = transcription_end_time - transcription_start_time
                transcription_speed = (video_length / transcription_time) * 100 if transcription_time > 0 else 0
                overall_confidence = overall_confidence * 100

                return {
                    "word_count": word_count,
                    "transcription_time": transcription_time,
                    "transcription_speed": transcription_speed,
                    "overall_confidence": overall_confidence
                }
            else:
                logger.error(f"Video Processing - Thread 2 - Audio Processing: Transcription job failed for video {video_id}")
                await status_tracker.update_process_status("transcription", "error", 0)
                await status_tracker.set_error("Transcription job failed.")
                return None

    except Exception as e:
        logger.error(f"Video Processing - Thread 2 - Audio Processing: Error transcribing audio for video {video_id}: {str(e)}", exc_info=True)
        await status_tracker.update_process_status("transcription", "error", 0)
        await status_tracker.set_error(f"Video Processing - Thread 2 - Audio Processing: Error transcribing audio: {str(e)}")
        return None
########################################################

## Processes the transcription
########################################################
async def process_transcription_response(video_id: str):
    plain_transcript = ""
    json_transcript = []
    word_count = 0
    total_confidence = 0

    try:
        # Find the transcript JSON file
        transcript_key = f"{video_id}/transcripts/audio_transcript_{video_id}.json"
        logger.debug(f"Attempting to retrieve transcript file for video {video_id}")
        try:
            async with get_s3_client() as s3_client:
                response = await s3_client.get_object(
                    Bucket=settings.PROCESSING_BUCKET, 
                    Key=transcript_key
                )
            transcript_content = await response['Body'].read()
            transcript_data = json.loads(transcript_content.decode('utf-8'))
            logger.debug(f"Successfully retrieved transcript file for video {video_id}")
        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"No transcript file found for video {video_id}")

        # Process the transcript data
        logger.debug(f"Processing transcript data for video {video_id}")
        items = transcript_data['results']['items']
        for item in items:
            if item['type'] == 'pronunciation':
                word = item['alternatives'][0]['content']
                confidence = float(item['alternatives'][0]['confidence'])
                start_time = item.get('start_time', '0')
                end_time = item.get('end_time', '0')

                word_count += 1
                total_confidence += confidence
                
                json_transcript.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "word": word,
                    "confidence": confidence
                })

                plain_transcript += word + " "
            elif item['type'] == 'punctuation':
                plain_transcript = plain_transcript.rstrip() + item['alternatives'][0]['content'] + " "
                # Add punctuation to the json_transcript as well
                if json_transcript and json_transcript[-1]['word']:  # Ensure that punctuation is applied to the last word
                    json_transcript[-1]['word'] += item['alternatives'][0]['content']


        # Clean up the plain transcript
        plain_transcript = plain_transcript.strip()

        # Calculate overall confidence score
        overall_confidence = total_confidence / word_count if word_count > 0 else 0

        logger.debug(f"Processed {word_count} words for video {video_id}")

        # Save transcript.json
        json_content = json.dumps(json_transcript, indent=2)
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/transcripts/transcript.json',
                Body=json_content,
                ContentType='application/json'
            )
        logger.debug(f"Saved JSON transcript for video {video_id}")

        # Save transcript.txt
        async with get_s3_client() as s3_client:
            await s3_client.put_object(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/transcripts/transcript.txt',
                Body=plain_transcript,
                ContentType='text/plain'
            )
        logger.debug(f"Saved plain text transcript for video {video_id}")

    except FileNotFoundError as e:
        logger.error(f"Transcript file not found for video {video_id}: {str(e)}")
        plain_transcript = "Transcript file not found."
        json_transcript = []
        word_count = 0
        overall_confidence = 0
        # Don't raise here, as this is a secondary operation
    except Exception as e:
        logger.error(f"Error processing transcription response for video {video_id}: {str(e)}", exc_info=True)
        plain_transcript = "Error processing transcription."
        json_transcript = []
        word_count = 0
        overall_confidence = 0
        # Don't raise here, as this is a secondary operation

    return plain_transcript, json_transcript, word_count, overall_confidence
########################################################

## Get the transcript
########################################################
async def get_transcript(video_id: str):
    transcript_key = f'{video_id}/transcripts/transcript.json'
    
    logger.debug(f"Attempting to retrieve transcript for video: {video_id}")
    
    try:
        async with get_s3_client() as s3_client:
            response = await s3_client.get_object(
                Bucket=settings.PROCESSING_BUCKET, 
                Key=transcript_key
            )
        transcript_data = await response['Body'].read()
        
        transcript = json.loads(transcript_data.decode('utf-8'))
        logger.debug(f"Successfully retrieved transcript for video: {video_id}")
        
        return transcript
    
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Transcript not found for video: {video_id}")
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding transcript JSON for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error decoding transcript")
    
    except Exception as e:
        logger.error(f"Error retrieving transcript for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving transcript")
########################################################