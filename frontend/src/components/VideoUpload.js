import React, { useState } from 'react';
import { Button, Typography, CircularProgress } from '@mui/material';
import { uploadVideo } from '../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [videoId, setVideoId] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    try {
      const response = await uploadVideo(file);
      setVideoId(response.video_id);
    } catch (error) {
      console.error('Error uploading video:', error);
    }
    setUploading(false);
  };

  return (
    <div>
      <input
        accept="video/*"
        style={{ display: 'none' }}
        id="raised-button-file"
        type="file"
        onChange={handleFileChange}
      />
      <label htmlFor="raised-button-file">
        <Button variant="contained" component="span">
          Choose File
        </Button>
      </label>
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleUpload} 
        disabled={!file || uploading}
        sx={{ ml: 2 }}
      >
        {uploading ? <CircularProgress size={24} /> : 'Upload Video'}
      </Button>
      {videoId && (
        <Typography sx={{ mt: 2 }}>
          Video uploaded successfully. Video ID: {videoId}
        </Typography>
      )}
    </div>
  );
};

export default VideoUpload;