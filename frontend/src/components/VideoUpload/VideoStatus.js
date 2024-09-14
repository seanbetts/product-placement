import React, { useState, useEffect, useCallback } from 'react';
import { Typography, CircularProgress, Box, Paper, Button, Divider } from '@mui/material';
import api from '../../services/api';

const VideoStatus = ({ videoId }) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pollingCount, setPollingCount] = useState(0);

  const fetchStatus = useCallback(async () => {
    try {
      const statusData = await api.getVideoStatus(videoId);
      setStatus(statusData);
      setLoading(false);
      // setError(null);  // Uncomment if you decide to use error state
  
      if (statusData.status === 'complete' || statusData.status === 'error') {
        return true; // Signal to stop polling
      }
      setPollingCount(prev => prev + 1);
    } catch (error) {
      console.error('Error fetching video status:', error);
      // setError('Failed to fetch status. Please try again.');  // Uncomment if you decide to use error state
      setLoading(false);
      return true; // Signal to stop polling on error
    }
    return false; // Continue polling
  }, [videoId]);

  useEffect(() => {
    let intervalId;
    const poll = async () => {
      const shouldStop = await fetchStatus();
      if (shouldStop) {
        clearInterval(intervalId);
      }
    };

    poll(); // Initial fetch
    intervalId = setInterval(poll, 5000); // Poll every 5 seconds

    return () => clearInterval(intervalId);
  }, [fetchStatus]);

  const handleRefresh = () => {
    setLoading(true);
    setPollingCount(0);
    fetchStatus();
  };

  if (loading) {
    return <CircularProgress />;
  }

  if (status?.status === 'error') {
    return (
      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography color="error">Error: {status.error || 'An unknown error occurred'}</Typography>
        <Button onClick={handleRefresh} sx={{ mt: 1 }}>Retry</Button>
      </Paper>
    );
  }

  const details = status?.details || {};

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Typography variant="h6" gutterBottom>Video Processing Status</Typography>
      <Typography>Video ID: {details.video_id}</Typography>
      <Typography>Status: {status?.status}</Typography>
      <Typography>Last Updated: {new Date(status?.last_updated).toLocaleString()}</Typography>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle1" gutterBottom>Video Details:</Typography>
      <Typography>Length: {details.video_length}</Typography>
      <Typography>Total Frames: {details.video?.total_frames}</Typography>
      <Typography>Extracted Frames: {details.video?.extracted_frames}</Typography>
      <Typography>Video FPS: {details.video?.video_fps}</Typography>
      <Typography>Processing Time: {details.video?.video_processing_time}</Typography>
      <Typography>Processing Speed: {details.video?.video_processing_speed}</Typography>
      <Typography>Processing FPS: {details.video?.video_processing_fps}</Typography>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle1" gutterBottom>Audio Details:</Typography>
      <Typography>Length: {details.audio?.audio_length}</Typography>
      <Typography>Processing Time: {details.audio?.audio_processing_time}</Typography>
      <Typography>Processing Speed: {details.audio?.audio_processing_speed}</Typography>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle1" gutterBottom>Transcription Details:</Typography>
      <Typography>Processing Time: {details.transcription?.transcription_processing_time}</Typography>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle1" gutterBottom>Total Processing:</Typography>
      <Typography>Start Time: {new Date(details.total_processing_start_time).toLocaleString()}</Typography>
      <Typography>End Time: {new Date(details.total_processing_end_time).toLocaleString()}</Typography>
      <Typography>Processing Time: {details.total_processing_time}</Typography>
      <Typography>Total Speed: {details.total_processing_speed}</Typography>

      {pollingCount > 60 && (
        <Box mt={2}>
          <Typography color="warning.main">Status updates stopped. The process might have completed.</Typography>
          <Button onClick={handleRefresh} sx={{ mt: 1 }}>Refresh Status</Button>
        </Box>
      )}
    </Paper>
  );
};

export default VideoStatus;