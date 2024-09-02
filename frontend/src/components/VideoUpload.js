import React, { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, CircularProgress, Box, Paper, Alert, Snackbar, Grid, LinearProgress, Stack, Divider } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CancelIcon from '@mui/icons-material/Cancel';
import PublishIcon from '@mui/icons-material/Publish';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import api from '../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedVideoId, setUploadedVideoId] = useState(null);
  const [cancelUpload, setCancelUpload] = useState(false);
  const cancelUploadRef = useRef(false);
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
    accept: {
      'video/mp4': ['.mp4'],
      'video/quicktime': ['.mov'],
      'video/x-msvideo': ['.avi'],
      'video/webm': ['.webm']
    },
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
    setCancelUpload(false);
    cancelUploadRef.current = false;
  
    try {
      const response = await api.uploadVideo(file, (progress) => {
        setUploadProgress(progress);
        if (cancelUpload) {
          throw new Error('Upload cancelled');
        }
      });
  
      setUploadedVideoId(response.video_id);
      setProcessingStatus({
        status: 'processing',
        progress: 0,
      });
      pollProcessingStatus(response.video_id);
    } catch (error) {
      console.error('Error uploading video:', error);
      if (error.message === 'Upload cancelled') {
        setError('Upload cancelled');
      } else {
        setError(error.response?.data?.detail || error.message || 'Failed to upload video. Please try again.');
      }
    } finally {
      setUploading(false);
      setCancelUpload(false);
      cancelUploadRef.current = false;
    }
  };

  const handleCancel = () => {
    if (uploading) {
      setCancelUpload(true);
      cancelUploadRef.current = true;
    } else {
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
      // Reset the cancelUpload state when not uploading
      setCancelUpload(false);
    }
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
      <Typography variant="caption" color="textSecondary">
        Accepted formats: .mp4, .mov, .avi, .webm
      </Typography>
    </Paper>
  );

  const renderFilePreview = () => (
    <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          {preview && (
            <Box sx={{ 
              width: '100%', 
              height: 0, 
              paddingBottom: '56.25%', // 16:9 aspect ratio
              position: 'relative', 
              overflow: 'hidden',
              borderRadius: 2,
              boxShadow: 3
            }}>
              <img 
                src={preview} 
                alt="Video preview" 
                style={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  width: '100%', 
                  height: '100%', 
                  objectFit: 'cover' 
                }} 
              />
            </Box>
          )}
          <Typography variant="subtitle1" sx={{ mt: 2, fontWeight: 'bold' }}>
            {file.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {(file.size / (1024 * 1024)).toFixed(2)} MB
          </Typography>
        </Grid>
        <Grid item xs={12} md={6}>
        {!uploading && !uploadedVideoId && (
          <Stack spacing={2} alignItems="center" sx={{ height: '100%', justifyContent: 'center' }}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleUpload} 
              startIcon={<PublishIcon />}
              size="large"
              fullWidth
            >
              Process Video
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<CancelIcon />}
              onClick={handleCancel}
              fullWidth
            >
              Cancel
            </Button>
          </Stack>
        )}
        {uploading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'center' }}>
            <CircularProgress size={60} thickness={5} />
            <Typography variant="h6" sx={{ mt: 2 }}>
              Uploading... {Math.round(uploadProgress)}%
            </Typography>
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<CancelIcon />}
              onClick={handleCancel}
              sx={{ mt: 2 }}
            >
              Cancel Upload
            </Button>
          </Box>
        )}
        {uploadedVideoId && renderProcessingStatus()}
      </Grid>
      </Grid>
    </Paper>
  );

  const renderProcessingStatus = () => (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h6" gutterBottom>Video Processing Status</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>Video ID: {uploadedVideoId}</Typography>
      <Divider sx={{ my: 2 }} />
      {renderProcessingProgress()}
      {processingStatus?.status === 'complete' && renderCompletedProcessing()}
    </Box>
  );

  const renderProgressBar = (label, status, progress) => (
    <Box sx={{ mt: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="body2">{label}</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {status === 'complete' ? (
            <CheckCircleOutlineIcon color="success" sx={{ mr: 1 }} />
          ) : (
            <HourglassEmptyIcon color="action" sx={{ mr: 1 }} />
          )}
          <Typography variant="body2" color="text.secondary">
            {status === 'complete' ? '100%' : `${Math.round(progress)}%`}
          </Typography>
        </Box>
      </Box>
      <LinearProgress 
        variant="determinate" 
        value={status === 'complete' ? 100 : progress} 
        sx={{ 
          height: 8, 
          borderRadius: 4,
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          '& .MuiLinearProgress-bar': {
            borderRadius: 4,
            backgroundColor: status === 'complete' ? 'success.main' : 'primary.main',
          }
        }}
      />
    </Box>
  );

  const renderProcessingProgress = () => (
    <Box>
      {renderProgressBar('Total Progress', processingProgress.total.status, processingProgress.total.progress)}
      {renderProgressBar('Video Processing', processingProgress.video.status, processingProgress.video.progress)}
      {renderProgressBar('Audio Processing', processingProgress.audio.status, processingProgress.audio.progress)}
      {renderProgressBar('Transcription', processingProgress.transcription.status, processingProgress.transcription.progress)}
      {renderProgressBar('Text Processing', processingProgress.ocr.status, processingProgress.ocr.progress)}
    </Box>
  );

  const renderCompletedProcessing = () => (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>Processing Complete</Typography>
      {processingStats ? (
        <>
          <Typography variant="body2">Processing Time: {processingStats.total_processing_time || 'N/A'}</Typography>
          <Typography variant="body2">Processing Speed: {processingStats.total_processing_speed || 'N/A'}</Typography>
        </>
      ) : (
        <Typography variant="body2">Fetching final stats...</Typography>
      )}
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleViewDetails}
        sx={{ mt: 2 }}
        fullWidth
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