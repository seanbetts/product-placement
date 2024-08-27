import React, { useState } from 'react';
import { Box } from '@mui/material';
import VideoUpload from './VideoUpload';
import VideoStatus from './VideoStatus';

const VideoUploadPage = () => {
  const [uploadedVideoId, setUploadedVideoId] = useState(null);

  const handleUploadSuccess = (videoId) => {
    console.log(`Video uploaded successfully with ID: ${videoId}`);
    setUploadedVideoId(videoId);
  };

  return (
    <Box>
      <VideoUpload onUploadSuccess={handleUploadSuccess} />
      {uploadedVideoId && <VideoStatus videoId={uploadedVideoId} />}
    </Box>
  );
};

export default VideoUploadPage;