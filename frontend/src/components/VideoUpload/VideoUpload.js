import React, { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { Button, Typography, CircularProgress, Box, Paper, Alert, Snackbar, Grid, LinearProgress, Stack, Divider, Chip } from '@mui/material';
import { red } from '@mui/material/colors';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CancelIcon from '@mui/icons-material/Cancel';
import PublishIcon from '@mui/icons-material/Publish';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import AssessmentIcon from '@mui/icons-material/Assessment';
import { styled, keyframes } from '@mui/material/styles';
import api from '../../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [uploadProgress, setUploadProgress] = useState(0);
  const [displayedUploadProgress, setDisplayedUploadProgress] = useState(0);
  const [extractingFrame, setExtractingFrame] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedVideoId, setUploadedVideoId] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [cancelUpload, setCancelUpload] = useState(false);
  const cancelUploadRef = useRef(false);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [processingStats, setProcessingStats] = useState(null);
  const [processingProgress, setProcessingProgress] = useState({
    total: { status: 'pending', progress: 0 },
    video: { status: 'pending', progress: 0 },
    audio: { status: 'pending', progress: 0 },
    transcription: { status: 'pending', progress: 0 },
    ocr: { status: 'pending', progress: 0 },
    objects: { status: 'pending', progress: 0 },
    annotation: { status: 'pending', progress: 0 }
  });
  const [videoDimensions, setVideoDimensions] = useState(null);
  const navigate = useNavigate();
  const [cancelSuccess, setCancelSuccess] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const videoFile = acceptedFiles[0];
    setFile(videoFile);
    setError(null);
    setExtractingFrame(true);
    try {
      const { frameUrl, width, height } = await extractFirstFrame(videoFile);
      setPreview(frameUrl);
      setVideoDimensions({ width, height });
    } catch (error) {
      console.error("Error extracting frame:", error);
      setError("Failed to generate video preview. The file might be corrupted or in an unsupported format.");
      setPreview(null);
      setVideoDimensions(null);
    } finally {
      setExtractingFrame(false);
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
          resolve({
            frameUrl: URL.createObjectURL(blob),
            width: video.videoWidth,
            height: video.videoHeight
          });
        }, 'image/jpeg', 0.75);
      };
      video.onerror = () => {
        reject(new Error("Error extracting video frame"));
      };
      video.src = URL.createObjectURL(file);
    });
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'pending':
        return 'Not Started';
      case 'in_progress':
        return 'In Progress';
      case 'complete':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return 'Not Started';
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setUploadProgress(0);
    setDisplayedUploadProgress(0);
    setError(null);
    setCancelUpload(false);
    cancelUploadRef.current = false;

    const cancelSignal = { isCancelled: false };

    try {
        const response = await api.uploadVideo(file, (progress) => {
            setUploadProgress(progress);
            if (!cancelUploadRef.current) {
                setDisplayedUploadProgress(progress);
            }
            if (cancelUploadRef.current) {
                cancelSignal.isCancelled = true;
            }
        }, cancelSignal);

        setUploadedVideoId(response.video_id);
        setProcessingStatus({
            status: 'processing',
            progress: 0,
        });
        // Start polling immediately after successful upload
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

const handleCancel = async () => {
    if (uploading) {
      setIsCancelling(true);
      setCancelUpload(true);
      cancelUploadRef.current = true;
      setDisplayedUploadProgress(0);  // Immediately reset displayed progress

      if (uploadedVideoId) {
        try {
          await api.cancelUpload(uploadedVideoId);
          setCancelSuccess(true);
        } catch (error) {
          console.error('Error cancelling upload:', error);
          setError('Failed to cancel upload. Please try again.');
        }
      } else {
        // Short delay to simulate cancellation process
        await new Promise(resolve => setTimeout(resolve, 500));
        setCancelSuccess(true);
      }
      
      setIsCancelling(false);
      setUploading(false);
    } else {
      // Reset all states
      setFile(null);
      setPreview(null);
      setUploadProgress(0);
      setDisplayedUploadProgress(0);
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
      setVideoDimensions(null);
      setCancelUpload(false);
    }
};

  const pollProcessingStatus = async (videoId) => {
    try {
      const response = await api.getVideoStatus(videoId);
      
      console.log('Polling response:', response);

      setProcessingProgress(prevProgress => ({
        ...prevProgress,
        total: { status: response.status, progress: response.progress || 0 },
        video_processing: { status: response.video_processing?.status || 'pending', progress: response.video_processing?.progress || 0 },
        audio_extraction: { status: response.audio_extraction?.status || 'pending', progress: response.audio_extraction?.progress || 0 },
        transcription: { status: response.transcription?.status || 'pending', progress: response.transcription?.progress || 0 },
        ocr: { status: response.ocr?.status || 'pending', progress: response.ocr?.progress || 0 },
        objects: { status: response.objects?.status || 'pending', progress: response.objects?.progress || 0 },
        annotation: { status: response.annotation?.status || 'pending', progress: response.annotation?.progress || 0 }
      }));

      setProcessingStatus({
        status: response.status,
        progress: response.progress || 0
      });

      if (response.status === 'complete') {
        fetchProcessingStats(videoId);
      } else {
        setTimeout(() => pollProcessingStatus(videoId), 1000);
      }
    } catch (error) {
      console.error('Error polling video status:', error);
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
          {extractingFrame ? (
            <Box sx={{ 
              width: '100%', 
              height: 0, 
              paddingBottom: '56.25%', // 16:9 aspect ratio
              position: 'relative', 
              overflow: 'hidden',
              borderRadius: 2,
              boxShadow: 3,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(0, 0, 0, 0.1)'
            }}>
              <Box sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <CircularProgress size={60} /> {/* Increased size for visibility */}
              </Box>
            </Box>
          ) : preview ? (
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
          ) : null}
          <Typography variant="h5" sx={{ mt: 2}}>
            {file.name}
          </Typography>
          {videoDimensions && (
            <Typography variant="body2" color="text.secondary">
              <strong>Resolution:</strong> {videoDimensions.width}x{videoDimensions.height}
            </Typography>
          )}
          <Typography variant="body2" color="text.secondary">
            <strong>File Size:</strong> {(file.size / (1024 * 1024)).toFixed(2)} MB
          </Typography>
          {processingStatus?.status === 'complete' && renderCompletedProcessing()}
        </Grid>
        <Grid item xs={12} md={6}>
        {!uploading && !uploadedVideoId && !isCancelling && (
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
        {(uploading || isCancelling) && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'center' }}>
            <CircularProgress 
              size={60} 
              thickness={5} 
              sx={{ 
                color: isCancelling ? red[500] : 'primary.main'
              }}
            />
            <Typography variant="h6" sx={{ mt: 2 }}>
              {isCancelling ? "Cancelling upload..." : `Uploading... ${Math.round(displayedUploadProgress)}%`}
            </Typography>
            {!isCancelling && (
              <Button
                variant="outlined"
                color="secondary"
                startIcon={<CancelIcon />}
                onClick={handleCancel}
                sx={{ mt: 2 }}
              >
                Cancel Upload
              </Button>
            )}
          </Box>
        )}
        {uploadedVideoId && renderProcessingStatus()}
      </Grid>
      </Grid>
    </Paper>
  );

  const renderProcessingStatus = () => (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h5" gutterBottom>Video Processing Status</Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom><strong>Video ID:</strong> {uploadedVideoId}</Typography>
      {renderProcessingProgress()}
    </Box>
  );

  // Create a rotation animation with pauses
  const spinWithPauses = keyframes`
    0% {
      transform: rotate(0deg);
    }
    25% {
      transform: rotate(180deg);
    }
    50% {
      transform: rotate(180deg);
    }
    75% {
      transform: rotate(360deg);
    }
    100% {
      transform: rotate(360deg);
    }
  `;

  // Create a styled component for the spinning icon with pauses
  const SpinningIcon = styled(HourglassEmptyIcon)(({ theme }) => ({
    animation: `${spinWithPauses} 4s cubic-bezier(0.65, 0, 0.35, 1) infinite`,
    color: theme.palette.primary.main,
  }));

  const renderProgressBar = (label, status, progress, isTotal = false) => (
    <Box sx={{ 
      mt: isTotal ? 3 : 2,
      mb: isTotal ? 3 : 2,
      p: isTotal ? 2 : 0,
      backgroundColor: isTotal ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
      borderRadius: 2,
    }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography 
          variant={isTotal ? "h6" : "body2"} 
          sx={{ fontWeight: isTotal ? 'bold' : 'normal' }}
        >
          {isTotal && <AssessmentIcon sx={{ mr: 1, verticalAlign: 'middle' }} />}
          {label}
        </Typography>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'flex-end'
        }}>
          {!isTotal && (
            <Chip
              label={getStatusBadge(status)}
              size="small"
              sx={{
                mr: 3,
                backgroundColor: status === 'complete' ? 'success.main' :
                                 status === 'error' ? 'error.main' :
                                 status === 'in_progress' ? 'primary.main' : 'grey.300',
                color: '#ffffff',
              }}
            />
          )}
          <Box sx={{ 
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            mr: isTotal ? 0 : 1
          }}>
            <Box sx={{ 
              width: isTotal ? '36px' : '24px',
              height: isTotal ? '36px' : '24px',
              display: 'flex', 
              justifyContent: 'center',
              alignItems: 'center',
              mr: isTotal ? 0.5 : 1
            }}>
              {status === 'complete' ? (
                <CheckCircleOutlineIcon sx={{ fontSize: isTotal ? 32 : 24 }} color="success" />
              ) : progress > 0 && progress < 100 ? (
                <SpinningIcon sx={{ fontSize: isTotal ? 32 : 24 }} />
              ) : (
                <HourglassEmptyIcon sx={{ 
                  fontSize: isTotal ? 32 : 24, 
                  color: progress === 0 ? 'grey.500' : 'primary.main' 
                }} />
              )}
            </Box>
            <Typography 
              variant={isTotal ? "h6" : "body2"} 
              color="text.secondary" 
              sx={{ 
                minWidth: isTotal ? '45px' : '40px',  // Changed from 60px to 45px for Total Progress
                textAlign: 'right',
                fontWeight: isTotal ? 'bold' : 'normal'
              }}
            >
              {status === 'complete' ? '100%' : `${Math.round(progress || 0)}%`}
            </Typography>
          </Box>
        </Box>
      </Box>
      <LinearProgress
        variant="determinate"
        value={status === 'complete' ? 100 : (progress || 0)}
        sx={{
          height: isTotal ? 12 : 8,
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
      {renderProgressBar('Total Progress', processingProgress.total?.status, processingProgress.total?.progress, true)}
      <Divider sx={{ my: 2 }} />
      {renderProgressBar('Extracting Video Frames...', processingProgress.video_processing?.status, processingProgress.video_processing?.progress)}
      {renderProgressBar('Extracting Audio...', processingProgress.audio_extraction?.status, processingProgress.audio_extraction?.progress)}
      {renderProgressBar('Transcribing Audio...', processingProgress.transcription?.status, processingProgress.transcription?.progress)}
      {renderProgressBar('Detecting Brands...', processingProgress.ocr?.status, processingProgress.ocr?.progress)}
      {renderProgressBar('Detecting Objects...', processingProgress.objects?.status, processingProgress.objects?.progress)}
      {renderProgressBar('Annotating Video...', processingProgress.annotation?.status, processingProgress.annotation?.progress)}
    </Box>
  );

  // <Box sx={{ 
  //   mt: isTotal ? 3 : 2,
  //   mb: isTotal ? 3 : 2,
  //   p: isTotal ? 2 : 0,
  //   backgroundColor: isTotal ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
  //   borderRadius: 2,
  // }}>

  const renderCompletedProcessing = () => (
    <Box sx={{ 
      mt: 3,
      p: 2,
      backgroundColor: 'rgba(0, 0, 0, 0.04)',
      borderRadius: 2,
    }}>
      <Typography variant="h5" gutterBottom>Processing Complete</Typography>
      {processingStats ? (
        <>
          <Typography variant="body2"><strong>Processing Time:</strong> {processingStats.total_processing_time ? parseFloat(processingStats?.total_processing_time).toFixed(0) : 'N/A'} seconds</Typography>
          <Typography variant="body2"><strong>Processing Speed:</strong> {processingStats.total_processing_speed || 'N/A'}</Typography>
        </>
      ) : (
        <Typography variant="body2">Fetching final stats...</Typography>
      )}
      <Button 
        variant="contained" 
        color="success" 
        onClick={handleViewDetails}
        sx={{
          mt: 2,
          maxWidth: '400px',  // Adjust this value as needed
          width: '100%',
          display: 'block',
          mx: 'auto'  // Centers the button
        }}
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

      <Snackbar
        open={cancelSuccess}
        autoHideDuration={6000}
        onClose={() => setCancelSuccess(false)}
      >
        <Alert
          onClose={() => setCancelSuccess(false)}
          severity="success"
          sx={{ width: '100%' }}
        >
          Upload cancelled successfully.
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default VideoUpload;