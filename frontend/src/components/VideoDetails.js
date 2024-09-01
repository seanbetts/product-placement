import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { 
  Typography, 
  Box, 
  CircularProgress,
  Divider,
  Grid, 
  Button, 
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TextField,
  InputAdornment,
  Tooltip,
  IconButton,
  Snackbar,
  Alert
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import {
  fetchVideoDetails,
  fetchVideoFrames,
  fetchTranscript,
  updateVideoName,
  downloadFile,
  setSearchTerm,
  setIsEditingName,
  setEditingName,
  setSnackbar
} from '../store/videoSlice';
import TextDetectionSection from './TextDetectionSection';

const VideoDetails = () => {
  const { videoId } = useParams();
  const dispatch = useDispatch();
  const { list: videos } = useSelector(state => state.videos);

  const video = videos.find(v => v.video_id === videoId);
  const videoDetails = useSelector(state => state.videos.details[videoId]);
  const frames = useSelector(state => state.videos.frames[videoId]?.data);
  const transcript = useSelector(state => state.videos.transcript[videoId]?.data);
  const searchTerm = useSelector(state => state.videos.searchTerm);
  const isEditingName = useSelector(state => state.videos.isEditingName);
  const editingName = useSelector(state => state.videos.editingName);
  const snackbar = useSelector(state => state.videos.snackbar);

  const [imagesLoaded, setImagesLoaded] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (!video) {
          await dispatch(fetchVideoDetails(videoId)).unwrap();
        }
        await dispatch(fetchVideoFrames(videoId)).unwrap();
        await dispatch(fetchTranscript(videoId)).unwrap();
      } catch (error) {
        console.error('Error fetching video data:', error);
        dispatch(setSnackbar({ open: true, message: 'Error fetching video data', severity: 'error' }));
      }
    };
    fetchData();
  }, [dispatch, videoId, video]);

  const sentenceTranscript = useMemo(() => {
    if (!Array.isArray(transcript) || transcript.length === 0) {
      return [];
    }

    let sentences = [];
    let currentSentence = { words: [], start_time: null, end_time: null, totalConfidence: 0, totalLength: 0 };

    transcript.forEach((word, index) => {
      if (!currentSentence.start_time) {
        currentSentence.start_time = word.start_time;
      }
      currentSentence.words.push(word.word);
      currentSentence.totalConfidence += word.confidence * word.word.length;
      currentSentence.totalLength += word.word.length;
      currentSentence.end_time = word.end_time;

      if (word.word.match(/[.!?]$/) || index === transcript.length - 1) {
        sentences.push({
          text: currentSentence.words.join(' '),
          start_time: currentSentence.start_time,
          end_time: currentSentence.end_time,
          confidence: currentSentence.totalConfidence / currentSentence.totalLength
        });
        currentSentence = { words: [], start_time: null, end_time: null, totalConfidence: 0, totalLength: 0 };
      }
    });

    return sentences;
  }, [transcript]);

  const filteredTranscript = useMemo(() => {
    if (!searchTerm) return sentenceTranscript;
    return sentenceTranscript.filter(sentence => 
      sentence?.text?.toLowerCase().includes(searchTerm?.toLowerCase())
    );
  }, [sentenceTranscript, searchTerm]);

  const handleNameEdit = () => {
    dispatch(setEditingName(videoName || videoId));
    dispatch(setIsEditingName(true));
  };  

  const handleNameChange = (event) => {
    dispatch(setEditingName(event.target.value));
  };

  const handleNameSubmit = async () => {
    if (editingName === videoDetails?.name) {
      dispatch(setIsEditingName(false));
      return;
    }
    setIsSubmitting(true);
    try {
      await dispatch(updateVideoName({ videoId, newName: editingName })).unwrap();
    } catch (error) {
      console.error('Error updating video name:', error);
      dispatch(setSnackbar({ open: true, message: 'Error updating video name', severity: 'error' }));
    }
    setIsSubmitting(false);
  };

  const handleNameCancel = () => {
    dispatch(setIsEditingName(false));
    dispatch(setEditingName(videoName || videoId));
  };

  const handleSearchChange = (event) => {
    dispatch(setSearchTerm(event.target.value));
  };

  const handleDownload = async (fileType) => {
    try {
      await dispatch(downloadFile({ videoId, fileType })).unwrap();
      // Handle successful download (e.g., open the file in a new tab)
    } catch (error) {
      console.error('Error downloading file:', error);
      dispatch(setSnackbar({ open: true, message: 'Error downloading file', severity: 'error' }));
    }
  };

  const handleImageLoad = (frameNumber) => {
    setImagesLoaded(prev => ({ ...prev, [frameNumber]: true }));
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    dispatch(setSnackbar({ ...snackbar, open: false }));
  };

  const formatDate = (isoString) => {
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
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const highlightText = (text, highlight) => {
    if (!highlight.trim()) {
      return <span>{text}</span>;
    }
    const regex = new RegExp(`(${highlight})`, 'gi');
    const parts = text.split(regex);
    return (
      <span>
        {parts.filter(String).map((part, i) => 
          regex.test(part) ? (
            <mark key={i} style={{ backgroundColor: 'yellow', padding: 0 }}>{part}</mark>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </span>
    );
  };

  const videoLoading = useSelector(state => state.videos.loading);
  const framesLoading = useSelector(state => state.videos.framesLoading);
  const transcriptLoading = useSelector(state => state.videos.transcriptLoading);
  const videoError = useSelector(state => state.videos.error);
  const framesError = useSelector(state => state.videos.framesError);
  const transcriptError = useSelector(state => state.videos.transcriptError);

  if (videoLoading || framesLoading || transcriptLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (videoError || framesError || transcriptError) {
    return (
      <Typography color="error" align="center">
        Error: {videoError || framesError || transcriptError}
      </Typography>
    );
  }

  if (!video) {
    return <Typography>No video details available</Typography>;
  }

  // Destructure video.details here
  const {
    name: videoName,
    video_length, 
    video: videoStats,
    audio, 
    transcription, 
    ocr, 
    total_processing_start_time,
    total_processing_end_time,
    total_processing_time = 'N/A',
    total_processing_speed = 'N/A'
  } = video.details || {};

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
            {videoName}
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
      
      {!isEditingName && videoName && (
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          ID: {videoId}
        </Typography>
      )}

      <Box sx={{ mt: 2, mb: 4 }}>
        <Typography variant="body1">
          Uploaded: {formatDate(total_processing_start_time ?? 'N/A')}
        </Typography>
        <Typography variant="body1">
          Processing Time: {total_processing_time ?? 'N/A'}
        </Typography>
        <Typography variant="body1">
          Video Length: {video_length ?? 'N/A'}
        </Typography>
        <Typography variant="body1">
          Frames Processed: {videoStats?.total_frames?.toLocaleString() ?? 'N/A'}
        </Typography>
      </Box>
      <Divider sx={{ my: 4 }} />

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Frames</Typography>
      <Box sx={{ overflowX: 'auto', whiteSpace: 'nowrap', mb: 4 }}>
        {frames && frames.length > 0 ? (
          frames.map((frame) => (
            <Box key={frame.number} sx={{ display: 'inline-block', mr: 2, position: 'relative' }}>
              {!imagesLoaded[frame.number] && (
                <Skeleton
                  variant="rectangular"
                  width={150}
                  height={150}
                  animation="wave"
                />
              )}
              <img 
                src={frame.url} 
                alt={`Frame from video ${videoId}`}
                style={{ 
                  height: '150px',
                  display: imagesLoaded[frame.number] ? 'block' : 'none'
                }} 
                onLoad={() => handleImageLoad(frame.number)}
              />
            </Box>
          ))
        ) : (
          <Typography>No frames available</Typography>
        )}
      </Box>

      <Divider sx={{ my: 4 }} />

      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Transcript</Typography>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search transcript..."
          value={searchTerm}
          onChange={handleSearchChange}
          sx={{ mb: 2 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        {filteredTranscript && filteredTranscript.length > 0 ? (
          <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
            <Table stickyHeader aria-label="transcript table">
              <TableHead>
                <TableRow>
                  <TableCell><Typography fontWeight="bold">Start Time</Typography></TableCell>
                  <TableCell><Typography fontWeight="bold">End Time</Typography></TableCell>
                  <TableCell><Typography fontWeight="bold">Sentence</Typography></TableCell>
                  <TableCell><Typography fontWeight="bold">Confidence</Typography></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredTranscript.map((sentence, index) => (
                  <TableRow key={index}>
                    <TableCell>{formatTime(parseFloat(sentence.start_time))}</TableCell>
                    <TableCell>{formatTime(parseFloat(sentence.end_time))}</TableCell>
                    <TableCell>{highlightText(sentence.text, searchTerm)}</TableCell>
                    <TableCell>{(sentence.confidence * 100).toFixed(2)}%</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography>No matching words found in the transcript</Typography>
        )}
      </Box>

      <Typography variant="h5" gutterBottom>Text Detection</Typography>
      <TextDetectionSection 
        videoId={videoId}
      />

      <Divider sx={{ my: 4 }} />

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Processing Stats</Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Video Stats</Typography>
            <Typography>Length: {video_length ?? 'N/A'}</Typography>
            <Typography>Total Frames: {videoStats?.total_frames?.toLocaleString() ?? 'N/A'}</Typography>
            <Typography>Extracted Frames: {videoStats?.extracted_frames?.toLocaleString() ?? 'N/A'}</Typography>
            <Typography>Video FPS: {videoStats?.video_fps ?? 'N/A'}</Typography>
            <Typography>Processing Time: {videoStats?.video_processing_time ?? 'N/A'}</Typography>
            <Typography>Processing Speed: {videoStats?.video_processing_speed ?? 'N/A'}</Typography>
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
            <Typography>Length: {audio?.audio_length ?? 'N/A'}</Typography>
            <Typography>Processing Time: {audio?.audio_processing_time ?? 'N/A'}</Typography>
            <Typography>Processing Speed: {audio?.audio_processing_speed ?? 'N/A'}</Typography>
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
            <Typography>Processing Time: {transcription?.transcription_processing_time ?? 'N/A'}</Typography>
            <Typography>Word Count: {transcription?.word_count?.toLocaleString() ?? 'N/A'}</Typography>
            <Typography>Confidence: {transcription?.confidence ?? 'N/A'}</Typography>
            <Typography>Transcription Speed: {transcription?.transcription_speed ?? 'N/A'}</Typography>
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
            {video?.ocr?.ocr_processing_time ? (
              <>
                <Typography>Processing Time: {ocr?.ocr_processing_time ?? 'N/A'}</Typography>
                <Typography>Frames Processed: {ocr?.frames_processed?.toLocaleString() ?? 'N/A'}</Typography>
                <Typography>Frames with Text: {ocr?.frames_with_text?.toLocaleString() ?? 'N/A'}</Typography>
                <Typography>Words Detected: {ocr?.total_words_detected?.toLocaleString() ?? 'N/A'}</Typography>
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
      <Typography>Start Time: {formatDate(total_processing_start_time)}</Typography>
      <Typography>End Time: {formatDate(total_processing_end_time)}</Typography>
      <Typography>Total Processing Time: {total_processing_time}</Typography>
      <Typography>Total Processing Speed: {total_processing_speed}</Typography>

      <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={handleSnackbarClose}>
        <Alert onClose={handleSnackbarClose} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default VideoDetails;