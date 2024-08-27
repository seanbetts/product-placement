import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { 
  Typography, 
  Box, 
  CircularProgress, 
  Grid, 
  Button, 
  Skeleton 
} from '@mui/material';
import api from '../services/api';

const VideoDetails = () => {
  const { videoId } = useParams();
  const [videoDetails, setVideoDetails] = useState(null);
  const [frames, setFrames] = useState([]);
  const [transcript, setTranscript] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imagesLoaded, setImagesLoaded] = useState({});

  useEffect(() => {
    const fetchVideoDetails = async () => {
      try {
        setLoading(true);
        const details = await api.getVideoDetails(videoId);
        setVideoDetails(details);

        const framesData = await api.getVideoFrames(videoId);
        setFrames(framesData);
        setImagesLoaded(framesData.reduce((acc, frame) => ({ ...acc, [frame.number]: false }), {}));

        const transcriptData = await api.getTranscript(videoId);
        setTranscript(transcriptData.transcript);

        setLoading(false);
      } catch (err) {
        console.error('Error fetching video details:', err);
        setError('Failed to load video details. Please try again later.');
        setLoading(false);
      }
    };

    fetchVideoDetails();
  }, [videoId]);

  const handleDownload = async (fileType) => {
    try {
      const response = await api.downloadFile(videoId, fileType);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${videoId}_${fileType}`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      } else {
        filename += getFileExtension(fileType);
      }
      
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(`Error downloading ${fileType}:`, error);
      // You might want to show an error message to the user here
    }
  };

  const getFileExtension = (fileType) => {
    switch(fileType) {
      case 'video':
        return '.mp4';
      case 'audio':
        return '.mp3';
      case 'transcript':
        return '.txt';
      default:
        return '';
    }
  };

  const handleImageLoad = (frameNumber) => {
    setImagesLoaded(prev => ({ ...prev, [frameNumber]: true }));
  };

  if (loading) return <CircularProgress />;
  if (error) return <Typography color="error">{error}</Typography>;
  if (!videoDetails) return <Typography>No video details available</Typography>;

  const details = videoDetails.details || {};
  const videoData = details.video || {};
  const audioData = details.audio || {};
  const transcriptionData = details.transcription || {};

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Video Details: {videoId}</Typography>

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Frames</Typography>
      <Box sx={{ overflowX: 'auto', whiteSpace: 'nowrap', mb: 4 }}>
        {frames.map((frame) => (
          <Box key={frame.number} sx={{ display: 'inline-block', mr: 2, position: 'relative' }}>
            {!imagesLoaded[frame.number] && (
              <Skeleton
                variant="rectangular"
                width={150}
                height={150}
                animation="wave"
              />
            )}
            <img 
              src={frame.url} 
              alt={`Frame from video ${videoId}`}
              style={{ 
                height: '150px',
                display: imagesLoaded[frame.number] ? 'block' : 'none'
              }} 
              onLoad={() => handleImageLoad(frame.number)}
            />
          </Box>
        ))}
      </Box>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Video Stats</Typography>
            <Typography>Length: {details.video_length || 'N/A'}</Typography>
            <Typography>Total Frames: {videoData.total_frames || 'N/A'}</Typography>
            <Typography>Extracted Frames: {videoData.extracted_frames || 'N/A'}</Typography>
            <Typography>Video FPS: {videoData.video_fps || 'N/A'}</Typography>
            <Typography>Processing Time: {videoData.video_processing_time || 'N/A'}</Typography>
            <Typography>Processing Speed: {videoData.video_processing_speed || 'N/A'}</Typography>
            <Box sx={{ flexGrow: 1 }} />
            <Button 
              variant="contained" 
              onClick={() => handleDownload('video')} 
              sx={{ mt: 2 }}
            >
              Download Video
            </Button>
          </Box>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Audio Stats</Typography>
            <Typography>Length: {audioData.audio_length || 'N/A'}</Typography>
            <Typography>Processing Time: {audioData.audio_processing_time || 'N/A'}</Typography>
            <Typography>Processing Speed: {audioData.audio_processing_speed || 'N/A'}</Typography>
            <Box sx={{ flexGrow: 1 }} />
            <Button 
              variant="contained" 
              onClick={() => handleDownload('audio')} 
              sx={{ mt: 2 }}
            >
              Download Audio
            </Button>
          </Box>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Transcript Stats</Typography>
            <Typography>Processing Time: {transcriptionData.transcription_processing_time || 'N/A'}</Typography>
            <Box sx={{ flexGrow: 1 }} />
            <Button 
              variant="contained" 
              onClick={() => handleDownload('transcript')} 
              sx={{ mt: 2 }}
            >
              Download Transcript
            </Button>
          </Box>
        </Grid>
      </Grid>

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Transcript</Typography>
      <Box sx={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #ccc', p: 2 }}>
        <Typography>{transcript}</Typography>
      </Box>
    </Box>
  );
};

export default VideoDetails;