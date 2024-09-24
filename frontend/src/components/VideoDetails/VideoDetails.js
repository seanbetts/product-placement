import React, { useEffect, useMemo, useState, useCallback, useTransition } from 'react';
import { useParams } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { 
  Typography, 
  Box, 
  CircularProgress,
  Divider,
  Grid, 
  Button, 
  TextField,
  InputAdornment,
  Tooltip,
  IconButton,
  Snackbar,
  Alert,
  useTheme,
  useMediaQuery
} from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import {
  fetchVideoDetails,
  fetchVideoFrames,
  updateVideoName,
  setSearchTerm,
  setIsEditingName,
  setEditingName,
  setSnackbar,
  selectVideoDetails,
  selectVideoFrames,
  selectVideoLoadingStates
} from '../../store/videoSlice';
import api from '../../services/api';
import { saveAs } from 'file-saver';
import { 
  fetchTranscript, 
  selectTranscript, 
  selectTranscriptLoadingState
} from '../../store/transcriptSlice';

const TextDetectionSection = React.lazy(() => import('./TextDetectionSection'));
const TranscriptTable = React.lazy(() => import('./TranscriptTable'));
const VideoFrames = React.lazy(() => import('./VideoFrames'));

const VideoDetails = () => {
  const theme = useTheme();
  const isLargeScreen = useMediaQuery(theme.breakpoints.up('md'));
  // eslint-disable-next-line
  const [isPending, startTransition] = useTransition();
  const { videoId } = useParams();
  const dispatch = useDispatch();
  
  const video = useSelector(state => selectVideoDetails(state, videoId));
  const frames = useSelector(state => selectVideoFrames(state, videoId));
  const { loadingDetails, loadingFrames } = useSelector(state => selectVideoLoadingStates(state, videoId));
  const transcript = useSelector(state => selectTranscript(state, videoId));
  const loadingTranscript = useSelector(state => selectTranscriptLoadingState(state, videoId));
  const searchTerm = useSelector(state => state.videos.ui.searchTerm);
  const isEditingName = useSelector(state => state.videos.ui.isEditingName);
  const editingName = useSelector(state => state.videos.ui.editingName);
  const snackbar = useSelector(state => state.videos.ui.snackbar);
  const error = useSelector(state => state.videos.status.error);

  const [initialLoading, setInitialLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const fetchData = useCallback(async () => {
    if (!videoId) {
      dispatch(setSnackbar({ open: true, message: 'Error: No video ID provided', severity: 'error' }));
      setInitialLoading(false);
      return;
    }
    try {
      if (!video && !loadingDetails && !frames && !loadingFrames && !transcript && !loadingTranscript) {
        startTransition(() => {
          Promise.all([
            dispatch(fetchVideoDetails(videoId)),
            dispatch(fetchVideoFrames(videoId)),
            dispatch(fetchTranscript(videoId))
          ]);
        });
      }
      setInitialLoading(false);
    } catch (error) {
      console.error('Error fetching video data:', error);
      dispatch(setSnackbar({ open: true, message: 'Error fetching video data', severity: 'error' }));
      setInitialLoading(false);
    }
  }, [dispatch, videoId, video, frames, transcript, loadingDetails, loadingFrames, loadingTranscript, startTransition]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleNameEdit = useCallback(() => {
    dispatch(setEditingName(video?.name || `${videoId}`));
    dispatch(setIsEditingName(true));
  }, [dispatch, video?.name, videoId]);

  const handleNameChange = useCallback((event) => {
    dispatch(setEditingName(event.target.value));
  }, [dispatch]);

  const handleNameSubmit = useCallback(async () => {
    if (editingName === video?.name) {
      dispatch(setIsEditingName(false));
      return;
    }
    setIsSubmitting(true);
    try {
      startTransition(() => {
        dispatch(updateVideoName({ videoId, newName: editingName })).unwrap();
      });
      dispatch(setSnackbar({ open: true, message: 'Video name updated successfully', severity: 'success' }));
    } catch (error) {
      console.error('Error updating video name:', error);
      dispatch(setSnackbar({ open: true, message: `Error updating video name: ${error.message || 'Unknown error'}`, severity: 'error' }));
    } finally {
      setIsSubmitting(false);
      dispatch(setIsEditingName(false));
    }
  }, [dispatch, editingName, video?.name, videoId, startTransition]);

  const handleNameCancel = useCallback(() => {
    dispatch(setIsEditingName(false));
    dispatch(setEditingName(video?.name || videoId));
  }, [dispatch, video?.name, videoId]);

  const handleSearchChange = useCallback((event) => {
    dispatch(setSearchTerm(event.target.value));
  }, [dispatch]);

  const handleSnackbarClose = useCallback((event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    dispatch(setSnackbar({ ...snackbar, open: false }));
  }, [dispatch, snackbar]);

  const formatDate = useCallback((isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date?.toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  }, []);

  const getFileExtension = useCallback((fileType) => {
    switch (fileType) {
      case 'video': return 'mp4';
      case 'audio': return 'mp3';
      case 'transcript': return 'txt';
      case 'word-cloud': return 'png';
      default: return 'txt';
    }
  }, []);

  const handleDownload = useCallback(async (fileType) => {
    try {
      const blob = await api.downloadFile(videoId, fileType);
      const fileName = `${video?.name || videoId}_${fileType}.${getFileExtension(fileType)}`;
      saveAs(blob, fileName);
    } catch (error) {
      console.error('Error downloading file:', error);
      dispatch(setSnackbar({ open: true, message: 'Error downloading file: ' + error.message, severity: 'error' }));
    }
  }, [videoId, video?.name, dispatch, getFileExtension]);

  const renderVideoPlaceholder = useMemo(() => {
    return (
      <Box
        sx={{
          width: '100%',
          paddingTop: '56.25%', // 16:9 Aspect Ratio
          position: 'relative',
          bgcolor: 'grey.300',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Typography
          variant="h6"
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        >
          Video Player Placeholder
        </Typography>
      </Box>
    );
  }, []);

  const renderVideoDetails = useMemo(() => {
    if (!video) return null;

    const {
      video_length = 'N/A', 
      video: videoStats = {},
      total_processing_start_time = null,
    } = video;

    return (
      <Box>
        <Typography variant="body1">
          <strong>Uploaded:</strong> {formatDate(total_processing_start_time)}
        </Typography>
        <Typography variant="body1">
          <strong>Video Length:</strong> {video_length}
        </Typography>
        <Typography variant="body1">
          <strong>Frames Processed:</strong> {videoStats?.total_frames?.toLocaleString() ?? 'N/A'}
        </Typography>
      </Box>
    );
  }, [video, formatDate]);

  const renderProcessingStats = useMemo(() => {
    if (!video) return null;

    const {
      video_length = 'N/A', 
      video: videoStats = {},
      audio = {}, 
      transcription = {}, 
      ocr = {}, 
      total_processing_start_time = null,
      total_processing_end_time = null,
      total_processing_time = 'N/A',
      total_processing_speed = 'N/A'
    } = video;

    return (
      <>
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Processing Stats</Typography>
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom>Video Stats</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Video Length:</Box>{' '}{video_length}</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Total Frames:</Box>{' '}{videoStats?.total_frames?.toLocaleString() ?? 'N/A'}</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Extracted Frames:</Box>{' '}{videoStats?.extracted_frames?.toLocaleString() ?? 'N/A'}</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Video FPS:</Box>{' '}{videoStats?.video_fps ? parseFloat(videoStats?.video_fps).toFixed(0) : 'N/A'}</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Processing Time:</Box>{' '}{videoStats?.video_processing_time ? parseFloat(videoStats?.video_processing_time).toFixed(0) : 'N/A'} seconds</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Processing Speed:</Box>{' '}{videoStats?.video_processing_speed ? parseFloat(videoStats?.video_processing_speed).toFixed(0) : 'N/A'}% of real-time</Typography>
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
          <Grid item xs={12} md={3}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom>Audio Stats</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Audio Length:</Box>{' '}{audio?.audio_length ? parseFloat(audio?.audio_length).toFixed(1) : 'N/A'} seconds</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Processing Time:</Box>{' '}{audio?.audio_processing_time ? parseFloat(audio?.audio_processing_time).toFixed(0) : 'N/A'} seconds</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Processing Speed:</Box>{' '}{audio?.audio_processing_speed ? parseFloat(audio?.audio_processing_speed).toFixed(0) : 'N/A'}% of real-time</Typography>
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
          <Grid item xs={12} md={3}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom>Transcript Stats</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Word Count:</Box>{' '}{transcription?.word_count?.toLocaleString() ?? 'N/A'}</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Processing Time:</Box>{' '}{transcription?.transcription_time ? parseFloat(transcription?.transcription_time).toFixed(0) : 'N/A'} seconds</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Confidence:</Box>{' '} {transcription?.overall_confidence  ? parseFloat(transcription?.overall_confidence).toFixed(1) : 'N/A'}%</Typography>
              <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Transcription Speed:</Box>{' '}{transcription?.transcription_speed ? parseFloat(transcription?.transcription_speed).toFixed(0) : 'N/A'}% of real-time</Typography>
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
          <Grid item xs={12} md={3}>
            <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom>Text Detection Stats</Typography>
              {ocr?.ocr_processing_time ? (
                <>
                  <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Frames Processed:</Box>{' '}{ocr?.frames_processed?.toLocaleString() ?? 'N/A'}</Typography>
                  <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Processing Time:</Box>{' '}{ocr?.ocr_processing_time  ? parseFloat(ocr?.ocr_processing_time).toFixed(0) : 'N/A'} seconds</Typography>
                  <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Frames with Text:</Box>{' '}{ocr?.frames_with_text?.toLocaleString() ?? 'N/A'}</Typography>
                  <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Words Detected:</Box>{' '}{ocr?.total_words_detected?.toLocaleString() ?? 'N/A'}</Typography>
                </>
              ) : (
                <Typography>Text Detection data not available</Typography>
              )}
              <Box sx={{ flexGrow: 1 }} />
              <Button 
                variant="contained" 
                onClick={() => handleDownload('word-cloud')} 
                sx={{ mt: 2 }}
              >
                Download Word Cloud
              </Button>
            </Box>
          </Grid>
        </Grid>
        
        <Typography variant="h6" gutterBottom>Total Processing Stats</Typography>
        <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Start Time:</Box>{' '}{formatDate(total_processing_start_time)}</Typography>
        <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>End Time:</Box>{' '}{formatDate(total_processing_end_time)}</Typography>
        <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Total Processing Time:</Box>{' '}{total_processing_time ? parseFloat(total_processing_time).toFixed(0) : 'N/A'} seconds</Typography>
        <Typography><Box component="span" sx={{ color: 'primary.main', fontWeight: 'bold' }}>Total Processing Speed:</Box>{' '}{total_processing_speed ? parseFloat(total_processing_speed).toFixed(0) : 'N/A'}% of real-time</Typography>
      </>
    );
  }, [video, formatDate, handleDownload]);

  if (initialLoading || loadingDetails) {
    return (
      <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Loading...</Typography>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Typography color="error" align="center">
        Error: {error}
      </Typography>
    );
  }
  
  if (!video) {
    return <Typography>No video details available</Typography>;
  }

  return (
    <Box sx={{ mt: 4 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="span" sx={{ whiteSpace: 'nowrap', mr: 1 }}>
          Video Details:
        </Typography>
        {isEditingName ? (
          <TextField
            value={editingName}
            onChange={handleNameChange}
            onKeyPress={(e) => e.key === 'Enter' && handleNameSubmit()}
            variant="standard"
            sx={{ flexGrow: 1 }}
            InputProps={{
              style: {
                fontSize: '2.125rem',
                fontWeight: 400,
                lineHeight: 1.235,
                letterSpacing: '0.00735em',
              },
              endAdornment: (
                <InputAdornment position="end">
                  <Tooltip title="Save">
                    <IconButton onClick={handleNameSubmit} size="small" disabled={isSubmitting}>
                      <SaveIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Cancel">
                    <IconButton onClick={handleNameCancel} size="small" disabled={isSubmitting}>
                      <CancelIcon />
                    </IconButton>
                  </Tooltip>
                </InputAdornment>
              ),
            }}
            autoFocus
          />
        ) : (
          <Typography 
            variant="h4" 
            component="span"
            sx={{ 
              cursor: 'pointer', 
              flexGrow: 1,
              '&:hover': { 
                '& .editIcon': { opacity: 1 } 
              }
            }}
            onDoubleClick={handleNameEdit}
          >
            {video.name || `${videoId}`}
            <Tooltip title="Edit name">
              <IconButton 
                onClick={handleNameEdit}
                size="small"
                sx={{ 
                  ml: 1, 
                  opacity: 0, 
                  transition: 'opacity 0.3s',
                  '&:hover': { opacity: 1 } 
                }}
                className="editIcon"
              >
                <EditIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Typography>
        )}
      </Box>
      
      {!isEditingName && video.name && (
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          ID: {videoId}
        </Typography>
      )}
      <Divider sx={{ my: 4 }} />
      <Grid container sx={{ mt: 2, mb: 4 }}>
        <Grid item xs={12} md={5.5} sx={{ pr: isLargeScreen ? 2 : 0, pb: isLargeScreen ? 0 : 2 }}>
          {renderVideoPlaceholder}
        </Grid>
        {isLargeScreen && (
          <Grid item sx={{ display: 'flex', justifyContent: 'center', width: '1px', margin: '15px' }}>
            <Divider orientation="vertical" flexItem />
          </Grid>
        )}
        <Grid item xs={12} md={5.5} sx={{ pl: isLargeScreen ? 2 : 0, pt: isLargeScreen ? 0 : 2 }}>
          {renderVideoDetails}
        </Grid>
      </Grid>

      <Divider sx={{ my: 4 }} />

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Video Frames</Typography>
      {loadingFrames ? (
        <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="200px">
          <CircularProgress />
          <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
            Loading...
          </Typography>
        </Box>
      ) : (
        <React.Suspense fallback={<CircularProgress />}>
          <VideoFrames
            frames={frames}
            videoId={videoId}
          />
        </React.Suspense>
      )}

      <Divider sx={{ my: 4 }} />

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Transcript</Typography>
      <React.Suspense fallback={
        <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="200px">
          <CircularProgress />
          <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Loading Transcript...</Typography>
        </Box>
      }>
        <TranscriptTable 
          videoId={videoId}
          transcript={transcript}
          searchTerm={searchTerm}
          onSearchChange={handleSearchChange}
        />
      </React.Suspense>

      <Divider sx={{ my: 4 }} />

      <Typography variant="h5" gutterBottom>Text Detection</Typography>
      <React.Suspense fallback={
        <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="200px">
          <CircularProgress />
          <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Loading Text Detection...</Typography>
        </Box>
      }>
        {video && video.ocr ? (
          <React.Suspense fallback={<CircularProgress />}>
            <TextDetectionSection videoId={videoId} ocrData={video.ocr} />
          </React.Suspense>
        ) : (
          <Typography>Text detection data not available</Typography>
        )}
      </React.Suspense>

      <Divider sx={{ my: 4 }} />

      {renderProcessingStats}

      <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={handleSnackbarClose}>
        <Alert onClose={handleSnackbarClose} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default React.memo(VideoDetails);