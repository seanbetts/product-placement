import React, { useMemo, useCallback, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  CircularProgress,
  TextField,
  InputAdornment,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { 
  fetchTranscript, 
  selectTranscript, 
  selectTranscriptLoadingState,
  selectTranscriptError,
  selectTranscriptErrorType
} from '../../store/transcriptSlice';

const TranscriptTable = React.memo(({ videoId, searchTerm, onSearchChange }) => {
  const dispatch = useDispatch();
  const transcript = useSelector(state => selectTranscript(state, videoId));
  const loading = useSelector(state => selectTranscriptLoadingState(state, videoId));
  const error = useSelector(state => selectTranscriptError(state, videoId));
  const errorType = useSelector(state => selectTranscriptErrorType(state, videoId));

  useEffect(() => {
    if (!transcript && !loading) {
      dispatch(fetchTranscript(videoId));
    }
  }, [dispatch, videoId, transcript, loading]);

  const formatTime = useCallback((seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }, []);

  const highlightText = useCallback((text, highlight) => {
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
  }, []);

  const processedTranscript = useMemo(() => {
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
    if (!searchTerm) return processedTranscript;
    return processedTranscript.filter(sentence => 
      sentence?.text?.toLowerCase().includes(searchTerm?.toLowerCase())
    );
  }, [processedTranscript, searchTerm]);

  if (loading) {
    return (
      <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Loading...</Typography>
      </Box>
    );
  }

  if (error) {
    let errorMessage;
    switch (errorType) {
      case 'NETWORK_ERROR':
        errorMessage = "Network error. Please check your connection.";
        break;
      case 'API_ERROR':
        errorMessage = "Server error. Please try again later.";
        break;
      default:
        errorMessage = `An unknown error occurred: ${error}`;
    }
    return <Typography color="error">{errorMessage}</Typography>;
  }

  if (!transcript || transcript.length === 0) {
    return <Typography>No transcript available</Typography>;
  }

  return (
    <>
      <TextField
        fullWidth
        variant="outlined"
        placeholder="Search transcript..."
        value={searchTerm}
        onChange={onSearchChange}
        sx={{ mb: 2 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
      />
      <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
        <Table stickyHeader aria-label="transcript table">
          <TableHead>
            <TableRow>
              <TableCell><Typography fontWeight="bold">Start</Typography></TableCell>
              <TableCell><Typography fontWeight="bold">End</Typography></TableCell>
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
                <TableCell>{(sentence.confidence * 100).toFixed(1)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
});

export default TranscriptTable;