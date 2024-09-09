import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { VariableSizeList as List } from 'react-window';
import { InView } from 'react-intersection-observer';

// Create a cache to store loaded images
const imageCache = new Map();

const Frame = React.memo(({ data, index, style }) => {
  const [imageState, setImageState] = useState(imageCache.has(data[index].url) ? 'loaded' : 'loading');
  const frame = data[index];
  const imgRef = useRef(null);

  useEffect(() => {
    if (imageCache.has(frame.url)) {
      setImageState('loaded');
      if (imgRef.current) {
        imgRef.current.src = imageCache.get(frame.url);
      }
    } else {
      setImageState('loading');
      const img = new Image();
      img.src = frame.url;
      img.onload = () => {
        imageCache.set(frame.url, frame.url);
        setImageState('loaded');
        if (imgRef.current) {
          imgRef.current.src = frame.url;
        }
      };
      img.onerror = () => {
        setImageState('error');
      };
    }
  }, [frame.url]);

  return (
    <Box style={style}>
      <InView triggerOnce>
        {({ inView, ref }) => (
          <Box ref={ref} sx={{ display: 'inline-block', mr: 2, position: 'relative' }}>
            {inView && (
              <>
                <img
                  ref={imgRef}
                  src={imageState === 'loaded' ? frame.url : ''}
                  alt={`Frame ${frame.number}`}
                  style={{
                    height: '150px',
                    width: '266px',
                    display: imageState === 'loaded' ? 'inline' : 'none',
                  }}
                />
                {imageState !== 'loaded' && (
                  <Box
                    sx={{
                      width: '266px',
                      height: '150px',
                      backgroundColor: '#f0f0f0',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                    }}
                  >
                    {imageState === 'loading' ? (
                      <CircularProgress size={40} />
                    ) : (
                      <Typography color="error">Error loading image</Typography>
                    )}
                  </Box>
                )}
              </>
            )}
          </Box>
        )}
      </InView>
    </Box>
  );
});

const VideoFrames = React.memo(({ frames, framesLoading, videoId }) => {
  const [containerWidth, setContainerWidth] = useState(0);
  const listRef = useRef(null);

  useEffect(() => {
    const updateWidth = () => {
      const container = document.getElementById('frames-container');
      if (container) {
        setContainerWidth(container.offsetWidth);
      }
    };

    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.resetAfterIndex(0);
    }
  }, [frames]);

  const getItemSize = useCallback(() => 266 + 16, []); // 266px width + 16px margin

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
        id="frames-container"
        sx={{
          overflowX: 'auto',
          overflowY: 'hidden',
          whiteSpace: 'nowrap',
          pb: 2,
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
            }}
          />
        ))}
      </Box>
    );
  }

  return (
    <Box
      id="frames-container"
      sx={{
        overflowX: 'hidden',
        overflowY: 'hidden',
        whiteSpace: 'nowrap',
        pb: 2,
      }}
    >
      <List
        ref={listRef}
        height={166} // 150px height + 16px padding
        itemCount={frames.length}
        itemSize={getItemSize}
        layout="horizontal"
        width={containerWidth}
        itemData={frames}
      >
        {Frame}
      </List>
    </Box>
  );
});

export default VideoFrames;