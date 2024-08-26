import React, { useState, useEffect } from 'react';
import { Typography, CircularProgress } from '@mui/material';
import { getVideoStatus } from '../services/api';

const VideoStatus = ({ videoId }) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const statusData = await getVideoStatus(videoId);
        setStatus(statusData);
      } catch (error) {
        console.error('Error fetching video status:', error);
      }
      setLoading(false);
    };

    fetchStatus();
    const intervalId = setInterval(fetchStatus, 5000); // Poll every 5 seconds

    return () => clearInterval(intervalId);
  }, [videoId]);

  if (loading) {
    return <CircularProgress />;
  }

  return (
    <div>
      <Typography variant="h6">Video Status</Typography>
      <Typography>Status: {status?.status}</Typography>
      <Typography>Progress: {status?.details?.progress}</Typography>
      {status?.status === 'complete' && (
        <div>
          <Typography>Processing Time: {status.details.processing_time}</Typography>
          <Typography>Extracted Frames: {status.details.extracted_frames}</Typography>
        </div>
      )}
    </div>
  );
};

export default VideoStatus;