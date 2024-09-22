import time
import json
import asyncio
from fastapi import HTTPException
from core.config import settings
from core.logging import AppLogger
from core.aws import get_s3_client
from models.status_tracker import StatusTracker
import boto3
from botocore.exceptions import ClientError

# Create a global instance of AppLogger
app_logger = AppLogger()

# Create a global instance of s3_client
s3_client = get_s3_client()

## Transcribes the video audio
########################################################
import asyncio
import time
import boto3
from botocore.exceptions import ClientError

async def transcribe_audio(vlogger, video_id: str, s3_client, video_length: float, status_tracker: StatusTracker):
    @vlogger.log_performance
    async def _transcribe_audio():
        vlogger.logger.info(f"Transcribing audio for video: {video_id}")
        audio_key = f'{video_id}/audio.mp3'
        transcript_key = f"{video_id}/transcripts/audio_transcript_{video_id}.json"

        try:
            # Check if the audio file exists
            try:
                await vlogger.log_performance(asyncio.to_thread)(
                    s3_client.head_object,
                    Bucket=settings.PROCESSING_BUCKET,
                    Key=audio_key
                )
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    vlogger.logger.warning(f"Audio file not found for video: {video_id}")
                    await status_tracker.update_process_status("transcription", "error", 0)
                    await status_tracker.set_error("Audio file not found for transcription.")
                    return None
                else:
                    raise

            # Instantiate the Transcribe client
            transcribe_client = boto3.client('transcribe')

            # Set up the transcription job
            job_name = f"audio_transcript_{video_id}"
            job_uri = f"s3://{settings.PROCESSING_BUCKET}/{audio_key}"
            
            transcription_start_time = time.time()
            vlogger.logger.info(f"Starting transcription job for video {video_id}")
            await vlogger.log_performance(asyncio.to_thread)(
                transcribe_client.start_transcription_job,
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
                status = await vlogger.log_performance(asyncio.to_thread)(
                    transcribe_client.get_transcription_job,
                    TranscriptionJobName=job_name
                )
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                
                await asyncio.sleep(5)
                elapsed_time = time.time() - transcription_start_time
                progress = min(95, (elapsed_time / timeout) * 100)  # Cap at 95% until completion
                await status_tracker.update_process_status("transcription", "in_progress", progress)
                vlogger.logger.debug(f"Transcription progress for video {video_id}: {progress:.2f}%")

            transcription_end_time = time.time()
            
            if job_status == 'COMPLETED':
                vlogger.logger.info(f"Transcription completed for video {video_id}")
                
                # Wait for the transcript file to be available in S3
                max_retries = 10
                for i in range(max_retries):
                    try:
                        await vlogger.log_performance(asyncio.to_thread)(
                            s3_client.head_object,
                            Bucket=settings.PROCESSING_BUCKET,
                            Key=transcript_key
                        )
                        vlogger.logger.info(f"Transcript file found for video {video_id}")
                        break
                    except ClientError:
                        if i < max_retries - 1:
                            await asyncio.sleep(5)
                        else:
                            raise FileNotFoundError(f"Transcript file not found for video {video_id} after {max_retries} retries")

                # Process the response and create transcripts
                plain_transcript, json_transcript, word_count, overall_confidence = await process_transcription_response(vlogger, s3_client, video_id)

                vlogger.logger.info(f"Transcripts processed and uploaded for video: {video_id}")

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
                vlogger.logger.error(f"Transcription job failed for video {video_id}")
                await status_tracker.update_process_status("transcription", "error", 0)
                await status_tracker.set_error("Transcription job failed.")
                return None

        except Exception as e:
            vlogger.logger.error(f"Error transcribing audio for video {video_id}: {str(e)}", exc_info=True)
            await status_tracker.update_process_status("transcription", "error", 0)
            await status_tracker.set_error(f"Error transcribing audio: {str(e)}")
            return None

    return await _transcribe_audio()
########################################################

## Processes the transcription
########################################################
async def process_transcription_response(vlogger, s3_client, video_id: str):
    @vlogger.log_performance
    async def _process_transcription_response():
        plain_transcript = ""
        json_transcript = []
        word_count = 0
        total_confidence = 0

        try:
            # Find the transcript JSON file
            transcript_key = f"{video_id}/transcripts/audio_transcript_{video_id}.json"
            vlogger.logger.info(f"Attempting to retrieve transcript file for video {video_id}")
            try:
                response = await vlogger.log_performance(s3_client.get_object)(
                    Bucket=settings.PROCESSING_BUCKET, 
                    Key=transcript_key
                )
                transcript_content = response['Body'].read()
                vlogger.log_s3_operation("download", len(transcript_content))
                transcript_data = json.loads(transcript_content.decode('utf-8'))
                vlogger.logger.info(f"Successfully retrieved transcript file for video {video_id}")
            except s3_client.exceptions.NoSuchKey:
                raise FileNotFoundError(f"No transcript file found for video {video_id}")

            # Process the transcript data
            vlogger.logger.info(f"Processing transcript data for video {video_id}")
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

            # Clean up the plain transcript
            plain_transcript = plain_transcript.strip()

            # Calculate overall confidence score
            overall_confidence = total_confidence / word_count if word_count > 0 else 0

            vlogger.logger.info(f"Processed {word_count} words for video {video_id}")

            # Save transcript.json
            json_content = json.dumps(json_transcript, indent=2)
            await vlogger.log_performance(s3_client.put_object)(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/transcripts/transcript.json',
                Body=json_content,
                ContentType='application/json'
            )
            vlogger.log_s3_operation("upload", len(json_content))
            vlogger.logger.info(f"Saved JSON transcript for video {video_id}")

            # Save transcript.txt
            await vlogger.log_performance(s3_client.put_object)(
                Bucket=settings.PROCESSING_BUCKET,
                Key=f'{video_id}/transcripts/transcript.txt',
                Body=plain_transcript,
                ContentType='text/plain'
            )
            vlogger.log_s3_operation("upload", len(plain_transcript))
            vlogger.logger.info(f"Saved plain text transcript for video {video_id}")

        except FileNotFoundError as e:
            vlogger.logger.error(f"Transcript file not found for video {video_id}: {str(e)}")
            plain_transcript = "Transcript file not found."
            json_transcript = []
            word_count = 0
            overall_confidence = 0
            # Don't raise here, as this is a secondary operation
        except Exception as e:
            vlogger.logger.error(f"Error processing transcription response for video {video_id}: {str(e)}", exc_info=True)
            plain_transcript = "Error processing transcription."
            json_transcript = []
            word_count = 0
            overall_confidence = 0
            # Don't raise here, as this is a secondary operation

        return plain_transcript, json_transcript, word_count, overall_confidence

    return await _process_transcription_response()
########################################################

## Get the transcript
########################################################
async def get_transcript(video_id: str):
    transcript_key = f'{video_id}/transcripts/transcript.json'
    
    # app_logger.log_info(f"Attempting to retrieve transcript for video: {video_id}")
    
    try:
        response = s3_client.get_object(
            Bucket=settings.PROCESSING_BUCKET, 
            Key=transcript_key
        )
        transcript_data = response['Body'].read()
        
        transcript = json.loads(transcript_data.decode('utf-8'))
        # app_logger.log_info(f"Successfully retrieved transcript for video: {video_id}")
        
        return transcript
    
    except s3_client.exceptions.NoSuchKey:
        app_logger.log_error(f"Transcript not found for video: {video_id}")
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    except json.JSONDecodeError as e:
        app_logger.log_error(f"Error decoding transcript JSON for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error decoding transcript")
    
    except Exception as e:
        app_logger.log_error(f"Error retrieving transcript for video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving transcript")
########################################################