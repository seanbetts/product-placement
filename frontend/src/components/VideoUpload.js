import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, CircularProgress, Box, Paper, Alert, Snackbar, Grid, LinearProgress, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CancelIcon from '@mui/icons-material/Cancel';
import PublishIcon from '@mui/icons-material/Publish';
import api from '../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedVideoId, setUploadedVideoId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [processingStats, setProcessingStats] = useState(null);
  const [processingProgress, setProcessingProgress] = useState({
    total: { status: 'pending', progress: 0 },
    video: { status: 'pending', progress: 0 },
    audio: { status: 'pending', progress: 0 },
    transcription: { status: 'pending', progress: 0 },
    ocr: { status: 'pending', progress: 0 }
  });
  const navigate = useNavigate();

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

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const response = await api.uploadVideo(file, (progress) => {
        setUploadProgress(progress);
      });
      setUploadedVideoId(response.video_id);
      setProcessingStatus({
        status: 'processing',
        progress: 0,
      });
      pollProcessingStatus(response.video_id);
    } catch (error) {
      console.error('Error uploading video:', error);
      setError(error.response?.data?.detail || 'Failed to upload video. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleCancel = () => {
    setFile(null);
    setPreview(null);
    setUploadProgress(0);
    setUploading(false);
    setUploadedVideoId(null);
    setProcessingStatus(null);
    setProcessingStats(null);
    setProcessingProgress({
      total: { status: 'pending', progress: 0 },
      video: { status: 'pending', progress: 0 },
      audio: { status: 'pending', progress: 0 },
      transcription: { status: 'pending', progress: 0 },
      ocr: { status: 'pending', progress: 0 }
    });
  };

  const pollProcessingStatus = async (videoId) => {
    try {
      const response = await api.getVideoStatus(videoId);
      
      setProcessingProgress(prevProgress => ({
        ...prevProgress,
        total: { status: response.status, progress: response.progress || 0 },
        video: { status: response.video_processing?.status || 'pending', progress: response.video_processing?.progress || 0 },
        audio: { status: response.audio_extraction?.status || 'pending', progress: response.audio_extraction?.progress || 0 },
        transcription: { status: response.transcription?.status || 'pending', progress: response.transcription?.progress || 0 },
        ocr: { status: response.ocr?.status || 'pending', progress: response.ocr?.progress || 0 }
      }));
  
      setProcessingStatus({
        status: response.status,
        progress: response.progress || 0
      });
  
      if (response.status === 'complete') {
        fetchProcessingStats(videoId);
      } else {
        // Increase polling frequency to 1 second (1000 milliseconds)
        setTimeout(() => pollProcessingStatus(videoId), 1000);
      }
    } catch (error) {
      console.error('Error polling video status:', error);
      // Even if there's an error, continue polling after a short delay
      setTimeout(() => pollProcessingStatus(videoId), 2000);
    }
  };

  const fetchProcessingStats = async (videoId) => {
    try {
      const stats = await api.getProcessingStats(videoId);
      setProcessingStats(stats);
    } catch (error) {
      console.error('Error fetching processing stats:', error);
      setProcessingStats(null); // Set to null in case of error
    }
  };

  const handleViewDetails = () => {
    navigate(`/video/${uploadedVideoId}`);
  };

  const renderUploadArea = () => (
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
  );

  const renderFilePreview = () => (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        {preview && (
          <Box sx={{ maxWidth: '100%', maxHeight: '360px', overflow: 'hidden' }}>
            <img src={preview} alt="Video preview" style={{ width: '100%', height: 'auto' }} />
          </Box>
        )}
        <Typography sx={{ mt: 1 }}>Selected file: {file.name}</Typography>
      </Grid>
      <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        {!uploading && !uploadedVideoId && (
          <Stack spacing={2} alignItems="center">
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleUpload} 
              startIcon={<PublishIcon />}
              size="large"
            >
              Process Video
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<CancelIcon />}
              onClick={handleCancel}
            >
              Cancel
            </Button>
          </Stack>
        )}
        {uploading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CircularProgress size={60} thickness={5} />
            <Typography variant="body2" sx={{ mt: 2 }}>
              Uploading... {Math.round(uploadProgress)}%
            </Typography>
          </Box>
        )}
        {uploadedVideoId && renderProcessingStatus()}
      </Grid>
    </Grid>
  );

  const renderProcessingStatus = () => (
    <Paper sx={{ p: 2, width: '100%' }}>
      <Typography variant="h6" gutterBottom>Video Processing Status</Typography>
      <Typography>Video ID: {uploadedVideoId}</Typography>
      <Typography>Status: {processingStatus?.status}</Typography>
      {processingStatus?.status === 'complete' 
        ? renderCompletedProcessing()
        : renderProcessingProgress()
      }
    </Paper>
  );

  const renderProgressBar = (label, status, progress) => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="body2">{label}: {status}</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Box sx={{ width: '100%', mr: 1 }}>
          <LinearProgress variant="determinate" value={progress} />
        </Box>
        <Box sx={{ minWidth: 35 }}>
          <Typography variant="body2" color="text.secondary">{`${Math.round(progress)}%`}</Typography>
        </Box>
      </Box>
    </Box>
  );

  const renderProcessingProgress = () => (
    <Box sx={{ mt: 2 }}>
      {renderProgressBar('Total Progress', processingProgress.total.status, processingProgress.total.progress)}
      {renderProgressBar('Video Processing', processingProgress.video.status, processingProgress.video.progress)}
      {renderProgressBar('Audio Processing', processingProgress.audio.status, processingProgress.audio.progress)}
      {renderProgressBar('Transcription', processingProgress.transcription.status, processingProgress.transcription.progress)}
      {renderProgressBar('Text Processing', processingProgress.ocr.status, processingProgress.ocr.progress)}
    </Box>
  );

  const renderCompletedProcessing = () => (
    <Box sx={{ mt: 2 }}>
      {processingStats ? (
        <>
          <Typography>Processing Time: {processingStats.total_processing_time || 'N/A'}</Typography>
          <Typography>Processing Speed: {processingStats.total_processing_speed || 'N/A'}</Typography>
        </>
      ) : (
        <Typography>Processing complete. Fetching final stats...</Typography>
      )}
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleViewDetails}
        sx={{ mt: 2 }}
      >
        View Processed Video
      </Button>
    </Box>
  );

  return (
    <Box sx={{ my: 4 }}>
      {!file && renderUploadArea()}
      {file && renderFilePreview()}

      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
      >
        <Alert 
          onClose={() => setError(null)} 
          severity="error" 
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default VideoUpload;