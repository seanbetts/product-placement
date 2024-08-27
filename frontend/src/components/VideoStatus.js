import React, { useState, useEffect, useCallback } from 'react';
import { Typography, CircularProgress, Box, Paper, Button } from '@mui/material';
import api from '../services/api';

const VideoStatus = ({ videoId }) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pollingCount, setPollingCount] = useState(0);

  const fetchStatus = useCallback(async () => {
    try {
      const statusData = await api.getVideoStatus(videoId);
      // ... rest of the function remains the same
    } catch (error) {
      // ... error handling remains the same
    }
    return false; // Continue polling
  }, [videoId, pollingCount]);

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

  if (error) {
    return (
      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography color="error">{error}</Typography>
        <Button onClick={handleRefresh} sx={{ mt: 1 }}>Retry</Button>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Typography variant="h6" gutterBottom>Video Processing Status</Typography>
      <Typography>Video ID: {videoId}</Typography>
      <Typography>Status: {status?.status}</Typography>
      
      {status?.status === 'processing' && status?.details && (
        <Box mt={2}>
          <Typography>Progress: {status.details.progress}</Typography>
          <Typography>Processing speed: {status.details.processing_speed}</Typography>
          <Typography>Frames Processed: {status.details.frames_processed}</Typography>
          <Typography>Estimated total time: {status.details.estimated_total_time}</Typography>
          <Typography>Estimated remaining time: {status.details.estimated_remaining_time}</Typography>
        </Box>
      )}

      {status?.status === 'complete' && status?.details && (
        <Box mt={2}>
          <Typography>Video Length: {status.details.video_length}</Typography>
          <Typography>Video FPS: {status.details.video_fps}</Typography>
          <Typography>Total Frames: {status.details.total_frames}</Typography>
          <Typography>Total Processing Time: {status.details.processing_time}</Typography>
          <Typography>Processing Speed: {status.details.processing_speed}</Typography>
          <Typography>
            Extracted Frames: {status.details.extracted_frames} 
            ({((status.details.extracted_frames / status.details.total_frames) * 100).toFixed(2)}%)
          </Typography>
        </Box>
      )}

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