import React, { useState, useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { 
  Typography, 
  Box, 
  CircularProgress, 
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
  InputAdornment
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import api from '../services/api';
import TextDetectionSection from './TextDetectionSection';

const VideoDetails = () => {
  const { videoId } = useParams();
  const [processingStats, setProcessingStats] = useState(null);
  const [frames, setFrames] = useState([]);
  const [transcript, setTranscript] = useState([]);
  const [processedOcrResults, setProcessedOcrResults] = useState([]);
  const [brandsOcrResults, setBrandsOcrResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [processedOcrError, setProcessedOcrError] = useState(null);
  const [brandsOcrError, setBrandsOcrError] = useState(null);
  const [imagesLoaded, setImagesLoaded] = useState({});
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchVideoDetails = async () => {
      try {
        setLoading(true);
        const stats = await api.getProcessingStats(videoId);
        setProcessingStats(stats);

        const framesData = await api.getVideoFrames(videoId);
        setFrames(framesData);
        setImagesLoaded(framesData.reduce((acc, frame) => ({ ...acc, [frame.number]: false }), {}));

        const transcriptData = await api.getTranscript(videoId);
        setTranscript(transcriptData);

        try {
          const processedOcrData = await api.getProcessedOcrResults(videoId);
          setProcessedOcrResults(processedOcrData);
        } catch (ocrErr) {
          console.error('Error fetching Processed Text Detection results:', ocrErr);
          setProcessedOcrError('Processed Text Detection results not available for this video.');
        }

        try {
          const brandsOcrData = await api.getBrandsOcrResults(videoId);
          setBrandsOcrResults(brandsOcrData);
        } catch (ocrErr) {
          console.error('Error fetching Brands Text Detection results:', ocrErr);
          setBrandsOcrError('Brands Text Detection results not available for this video.');
        }

        setLoading(false);
      } catch (err) {
        console.error('Error fetching video details:', err);
        setError('Failed to load video details. Please try again later.');
        setLoading(false);
      }
    };

    fetchVideoDetails();
  }, [videoId]);

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
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


  const sentenceTranscript = useMemo(() => {
    if (!Array.isArray(transcript) || transcript.length === 0) {
      return [];
    }

    let sentences = [];
    let currentSentence = { words: [], start_time: null, end_time: null, totalConfidence: 0, totalLength: 0 };

    transcript?.forEach((word, index) => {
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

  const filteredTranscript = useMemo(() => {
    if (!searchTerm) return sentenceTranscript;
    return sentenceTranscript.filter(sentence => 
      sentence?.text?.toLowerCase().includes(searchTerm?.toLowerCase())
    );
  }, [sentenceTranscript, searchTerm]);

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleDownload = async (fileType) => {
    try {
      const response = await api.downloadFile(videoId, fileType);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${videoId}_${fileType}`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      } else {
        filename += getFileExtension(fileType);
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(`Error downloading ${fileType}:`, error);
    }
  };

  const getFileExtension = (fileType) => {
    switch(fileType) {
      case 'video':
        return '.mp4';
      case 'audio':
        return '.mp3';
      case 'transcript':
        return '.txt';
      default:
        return '';
    }
  };

  const handleImageLoad = (frameNumber) => {
    setImagesLoaded(prev => ({ ...prev, [frameNumber]: true }));
  };

  if (loading) return <CircularProgress />;
  if (error) return <Typography color="error">{error}</Typography>;
  if (!processingStats) return <Typography>No video details available</Typography>;

  const {
    video_length,
    video,
    audio,
    transcription,
    ocr,
    total_processing_start_time,
    total_processing_end_time,
    total_processing_time,
    total_processing_speed
  } = processingStats;

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>Video Details: {videoId}</Typography>
      {total_processing_end_time && (
        <Typography variant="subtitle1" gutterBottom color="text.secondary">
          Processed on: {formatDate(total_processing_end_time)}
        </Typography>
      )}

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Frames</Typography>
      <Box sx={{ overflowX: 'auto', whiteSpace: 'nowrap', mb: 4 }}>
        {frames.map((frame) => (
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
        ))}
      </Box>

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
        {filteredTranscript.length > 0 ? (
          <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
            <Table stickyHeader aria-label="transcript table">
              <TableHead>
                <TableRow>
                  <TableCell>Start Time</TableCell>
                  <TableCell>End Time</TableCell>
                  <TableCell>Sentence</TableCell>
                  <TableCell>Confidence</TableCell>
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
          <Typography>No matching transcript found</Typography>
        )}
      </Box>

      <TextDetectionSection 
        videoId={videoId}
        processedOcrResults={processedOcrResults}
        brandsOcrResults={brandsOcrResults}
        videoFps={video.video_fps}
      />

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>Processing Stats</Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Video Stats</Typography>
            <Typography>Length: {video_length || 'N/A'}</Typography>
            <Typography>Total Frames: {video?.total_frames?.toLocaleString() ?? 'N/A'}</Typography>
            <Typography>Extracted Frames: {video?.extracted_frames?.toLocaleString() ?? 'N/A'}</Typography>
            <Typography>Video FPS: {video.video_fps || 'N/A'}</Typography>
            <Typography>Processing Time: {video.video_processing_time || 'N/A'}</Typography>
            <Typography>Processing Speed: {video.video_processing_speed || 'N/A'}</Typography>
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
            <Typography>Length: {audio.audio_length || 'N/A'}</Typography>
            <Typography>Processing Time: {audio.audio_processing_time || 'N/A'}</Typography>
            <Typography>Processing Speed: {audio.audio_processing_speed || 'N/A'}</Typography>
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
            <Typography>Processing Time: {transcription.transcription_processing_time || 'N/A'}</Typography>
            <Typography>Word Count: {transcription?.word_count?.toLocaleString() ?? 'N/A'}</Typography>
            <Typography>Confidence: {transcription.confidence || 'N/A'}</Typography>
            <Typography>Transcription Speed: {transcription.transcription_speed || 'N/A'}</Typography>
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
            {ocr.ocr_processing_time ? (
              <>
                <Typography>Processing Time: {ocr?.ocr_processing_time || 'N/A'}</Typography>
                <Typography>Frames Processed: {ocr?.frames_processed?.toLocaleString() ?? 'N/A'}</Typography>
                <Typography>Frames with Text: {ocr?.frames_with_text?.toLocaleString() ?? 'N/A'}</Typography>
                <Typography>Words Detected: {ocr?.total_words_detected?.toLocaleString() ?? 'N/A'}</Typography>
              </>
            ) : (
              <Typography>Text Detection data not available</Typography>
            )}
          </Box>
        </Grid>
      </Grid>
      
      <Typography variant="h6" gutterBottom>Total Processing Stats</Typography>
      <Typography>Start Time: {formatDate(total_processing_start_time)}</Typography>
      <Typography>End Time: {formatDate(total_processing_end_time)}</Typography>
      <Typography>Total Processing Time: {total_processing_time}</Typography>
      <Typography>Total Processing Speed: {total_processing_speed}</Typography>
    </Box>
  );
};

export default VideoDetails;