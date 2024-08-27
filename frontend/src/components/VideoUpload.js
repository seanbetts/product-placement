import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button, Typography, CircularProgress, Box, Paper, Alert, Snackbar } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import api from '../services/api';

const VideoUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedVideoId, setUploadedVideoId] = useState(null);

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
      if (typeof onUploadSuccess === 'function') {
        onUploadSuccess(response.video_id);
      }
    } catch (error) {
      console.error('Error uploading video:', error);
      setError(error.response?.data?.detail || 'Failed to upload video. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleCloseError = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setError(null);
  };

  return (
    <Box sx={{ my: 4 }}>
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
      
      {file && (
        <Box sx={{ mt: 2 }}>
          <Typography>Selected file: {file.name}</Typography>
          {preview && (
            <Box sx={{ mt: 2, maxWidth: '640px', maxHeight: '360px', overflow: 'hidden' }}>
              <img src={preview} alt="Video preview" style={{ width: '100%', height: 'auto' }} />
            </Box>
          )}
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleUpload} 
            disabled={uploading}
            sx={{ mt: 2 }}
          >
            {uploading ? <CircularProgress size={24} /> : 'Upload Video'}
          </Button>
        </Box>
      )}

      {uploadedVideoId && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Video uploaded successfully! Video ID: {uploadedVideoId}
        </Alert>
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