import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { keyframes } from '@mui/system';

const shimmer = keyframes`
  0% { background-position: -468px 0; }
  100% { background-position: 468px 0; }
`;

const VideoFrames = React.memo(({ frames, framesLoading, videoId }) => {
  if (framesLoading) {
    return (
      <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Loading...</Typography>
      </Box>
    );
  }

  if (!frames || frames.length === 0) {
    return (
      <Box
        sx={{
          overflowX: 'auto',
          overflowY: 'hidden',
          whiteSpace: 'nowrap',
          pb: 2,
        }}
      >
        <Box
          sx={{
            display: 'inline-block',
            minWidth: '100%',
          }}
        >
          {[...Array(5)].map((_, index) => (
            <Box 
              key={index} 
              sx={{ 
                display: 'inline-block', 
                mr: 2,
                width: '266px',
                height: '150px',
                backgroundColor: '#f0f0f0',
                backgroundImage: 'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)',
                backgroundSize: '200% 100%',
                animation: `${shimmer} 1.5s infinite`,
              }}
            />
          ))}
        </Box>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        overflowX: 'auto',
        overflowY: 'hidden',
        whiteSpace: 'nowrap',
        pb: 2,
      }}
    >
      <Box
        sx={{
          display: 'inline-block',
          minWidth: '100%',
        }}
      >
        {frames.map((frame) => (
          <Box 
            key={frame.number} 
            sx={{ 
              display: 'inline-block', 
              mr: 2, 
            }}
          >
            <img 
              src={frame.url} 
              alt={`Frame ${frame.number} from video ${videoId}`}
              style={{ 
                height: '150px',
              }} 
            />
          </Box>
        ))}
      </Box>
    </Box>
  );
});

export default VideoFrames;