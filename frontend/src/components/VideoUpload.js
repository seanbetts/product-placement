import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, CircularProgress, Box, Paper, Alert, Snackbar, Grid, LinearProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import api from '../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedVideoId, setUploadedVideoId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [processingStats, setProcessingStats] = useState(null);
  const navigate = useNavigate();

  const extractFirstFrame = (file) => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadeddata = () => {
        video.currentTime = 0;
      };
      video.onseeked = () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          resolve(URL.createObjectURL(blob));
        }, 'image/jpeg', 0.75);
      };
      video.onerror = () => {
        reject(new Error("Error extracting video frame"));
      };
      video.src = URL.createObjectURL(file);
    });
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const videoFile = acceptedFiles[0];
    setFile(videoFile);
    setError(null);
    try {
      const frameUrl = await extractFirstFrame(videoFile);
      setPreview(frameUrl);
    } catch (error) {
      console.error("Error extracting frame:", error);
      setError("Failed to generate video preview. The file might be corrupted or in an unsupported format.");
      setPreview(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: 'video/*',
    multiple: false
  });

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const response = await api.uploadVideo(file);
      setUploadedVideoId(response.video_id);
      setProcessingStatus({
        status: 'processing',
        progress: 0,
        lastUpdated: new Date().toLocaleString()
      });
      pollProcessingStatus(response.video_id);
    } catch (error) {
      console.error('Error uploading video:', error);
      setError(error.response?.data?.detail || 'Failed to upload video. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const pollProcessingStatus = async (videoId) => {
    try {
      const response = await api.getVideoStatus(videoId);
      setProcessingStatus({
        status: response.status,
        progress: response.progress || 0,
        lastUpdated: new Date(response.last_updated).toLocaleString()
      });
      if (response.status === 'complete') {
        fetchProcessingStats(videoId);
      } else {
        setTimeout(() => pollProcessingStatus(videoId), 5000);
      }
    } catch (error) {
      console.error('Error polling video status:', error);
    }
  };

  const fetchProcessingStats = async (videoId) => {
    try {
      const stats = await api.getProcessingStats(videoId);
      setProcessingStats(stats);
    } catch (error) {
      console.error('Error fetching processing stats:', error);
    }
  };

  const handleCloseError = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setError(null);
  };

  const handleViewDetails = () => {
    navigate(`/video/${uploadedVideoId}`);
  };

  const renderProcessingStats = () => {
    if (!processingStats) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6">Processing Statistics</Typography>
        <Typography>Video Length: {processingStats.video_length}</Typography>
        <Typography>Total Processing Time: {processingStats.total_processing_time}</Typography>
        <Typography>Total Processing Speed: {processingStats.total_processing_speed}</Typography>
        <Typography>Video Processing Time: {processingStats.video.video_processing_time}</Typography>
        <Typography>Audio Processing Time: {processingStats.audio.audio_processing_time}</Typography>
        <Typography>Transcription Processing Time: {processingStats.transcription.transcription_processing_time}</Typography>
        <Typography>OCR Processing Time: {processingStats.ocr.ocr_processing_time}</Typography>
        <Typography>Frames Processed: {processingStats.ocr.frames_processed}</Typography>
        <Typography>Frames with Text: {processingStats.ocr.frames_with_text}</Typography>
      </Box>
    );
  };

  return (
    <Box sx={{ my: 4 }}>
      {!uploadedVideoId && (
        <Paper
          {...getRootProps()}
          sx={{
            p: 3,
            border: '2px dashed #ccc',
            borderRadius: 2,
            textAlign: 'center',
            cursor: 'pointer',
            bgcolor: isDragActive ? 'action.hover' : 'background.paper',
            transition: 'all 0.3s ease',
            '&:hover': {
              borderColor: 'primary.main',
              bgcolor: 'rgba(0, 0, 0, 0.04)'
            }
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 48, mb: 2, color: 'primary.main' }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? "Drop the video here" : "Drag 'n' drop a video here"}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            or click to select file
          </Typography>
        </Paper>
      )}

      {file && (
        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} md={6}>
            {preview && (
              <Box sx={{ maxWidth: '100%', maxHeight: '360px', overflow: 'hidden' }}>
                <img src={preview} alt="Video preview" style={{ width: '100%', height: 'auto' }} />
              </Box>
            )}
            <Typography sx={{ mt: 1 }}>Selected file: {file.name}</Typography>
          </Grid>
          {uploadedVideoId ? (
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>Video Processing Status</Typography>
                <Typography>Video ID: {uploadedVideoId}</Typography>
                <Typography>Status: {processingStatus?.status}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress variant="determinate" value={processingStatus?.progress || 0} />
                  </Box>
                  <Box sx={{ minWidth: 35 }}>
                    <Typography variant="body2" color="text.secondary">{`${Math.round(
                      processingStatus?.progress || 0,
                    )}%`}</Typography>
                  </Box>
                </Box>
                <Typography>Last Updated: {processingStatus?.lastUpdated}</Typography>
                {processingStatus?.status === 'complete' && renderProcessingStats()}
                {processingStatus?.status === 'complete' && (
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={handleViewDetails}
                    sx={{ mt: 2 }}
                  >
                    View Processed Video
                  </Button>
                )}
              </Paper>
            </Grid>
          ) : (
            <Grid item xs={12} md={6} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleUpload} 
                disabled={uploading}
              >
                {uploading ? <CircularProgress size={24} /> : 'Process Video'}
              </Button>
            </Grid>
          )}
        </Grid>
      )}

      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseError}>
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default VideoUpload;