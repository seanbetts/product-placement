import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  CircularProgress,
  Card,
  CardMedia,
  CardContent,
  Grid,
  Divider
} from '@mui/material';
import api from '../services/api';

const VideoHistory = () => {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const data = await api.getProcessedVideos();
        console.log('Fetched videos:', data);
        setVideos(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching processed videos:', err);
        setError('Failed to load video history. Please try again later.');
        setLoading(false);
      }
    };

    fetchVideos();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" align="center">
        {error}
      </Typography>
    );
  }

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom component="div" sx={{ mb: 3 }}>
        Processed Videos History
      </Typography>
      {videos.length === 0 ? (
        <Typography>No processed videos found.</Typography>
      ) : (
        <Grid container spacing={3}>
          {videos.map((video) => (
            <Grid item xs={12} sm={6} md={4} key={video.video_id}>
              <Card>
                <CardMedia
                  component="img"
                  height="140"
                  image={`/video-frame/${video.video_id}`}
                  alt={`First frame of video ${video.video_id}`}
                />
                <CardContent>
                  <Typography variant="h6" component="div" noWrap>
                    {video.video_id}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Length: {video.video_length}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Video:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Frames: {video.video.total_frames}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Extracted Frames: {video.video.extracted_frames}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time: {video.video.video_processing_time}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Speed: {video.video.video_processing_speed}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Audio:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Length: {video.audio.audio_length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time: {video.audio.audio_processing_time}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Speed: {video.audio.audio_processing_speed}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Total:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Speed: {video.total_processing_speed}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default VideoHistory;