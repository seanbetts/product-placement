import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Typography,
  Box,
  CircularProgress,
  Card,
  CardMedia,
  CardContent,
  Grid,
  Collapse,
  Button,
  Skeleton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Divider,
  InputAdornment,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ClearIcon from '@mui/icons-material/Clear';
import SearchIcon from '@mui/icons-material/Search';
import { useNavigate } from 'react-router-dom';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { format } from 'date-fns';
import { fetchProcessedVideos, fetchFirstVideoFrame } from '../../store/videoSlice';

const VideoHistory = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const videos = useSelector(state => state.videos.data.list);
  const firstFrames = useSelector(state => state.videos.data.firstFrames);
  const loading = useSelector(state => state.videos.status.loading);
  const error = useSelector(state => state.videos.status.error);
  
  const [expanded, setExpanded] = useState({});
  const [imagesLoaded, setImagesLoaded] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [sortCriteria, setSortCriteria] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');
  const [firstFramesLoading, setFirstFramesLoading] = useState({});
  const processedVideos = useRef(new Set());
  const hasCheckedVideos = useRef(false);

  const filterAndSortVideos = useCallback((videos, searchTerm, startDate, endDate, sortCriteria, sortOrder) => {
    if (!Array.isArray(videos) || videos.length === 0) {
      return [];
    }

    let result = videos.filter(video => {
      const videoDate = new Date(video.details.total_processing_end_time);
      videoDate.setHours(0, 0, 0, 0);

      return (video.details.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        video.video_id.toLowerCase().includes(searchTerm.toLowerCase())) &&
        (!startDate || videoDate >= startDate) &&
        (!endDate || videoDate <= endDate);
    });

    result.sort((a, b) => {
      let comparison = 0;
      switch (sortCriteria) {
        case 'date':
          comparison = new Date(b.details.total_processing_end_time).getTime() - new Date(a.details.total_processing_end_time).getTime();
          break;
        case 'length':
          comparison = parseFloat(b.details.video_length) - parseFloat(a.details.video_length);
          break;
        default:
          comparison = 0;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return result;
  }, []);

  const filteredVideos = useMemo(() => 
    filterAndSortVideos(videos, searchTerm, startDate, endDate, sortCriteria, sortOrder),
    [videos, searchTerm, startDate, endDate, sortCriteria, sortOrder, filterAndSortVideos]
  );

  const memoizedDispatchFirstFrames = useCallback(() => {
    videos.forEach(video => {
      if (!firstFrames[video.video_id] && !firstFramesLoading[video.video_id] && !processedVideos.current.has(video.video_id)) {
        setFirstFramesLoading(prev => ({ ...prev, [video.video_id]: true }));
        processedVideos.current.add(video.video_id);
        dispatch(fetchFirstVideoFrame(video.video_id))
          .then(() => {
            setFirstFramesLoading(prev => ({ ...prev, [video.video_id]: false }));
          })
          .catch(error => {
            console.error(`Error fetching first frame for video ${video.video_id}:`, error);
            setFirstFramesLoading(prev => ({ ...prev, [video.video_id]: false }));
          });
      }
    });
  }, [videos, firstFrames, dispatch, firstFramesLoading]);

  useEffect(() => {
    if (!hasCheckedVideos.current) {
      dispatch(fetchProcessedVideos());
      hasCheckedVideos.current = true;
    }
  }, [dispatch]);

  useEffect(() => {
    memoizedDispatchFirstFrames();
  }, [memoizedDispatchFirstFrames]);

  useEffect(() => {
    if (Array.isArray(videos) && videos.length > 0) {
      setExpanded(videos.reduce((acc, video) => ({ ...acc, [video.video_id]: false }), {}));
      setImagesLoaded(videos.reduce((acc, video) => ({ ...acc, [video.video_id]: false }), {}));
    }
  }, [videos]);

  const handleExpandClick = (videoId) => {
    setExpanded(prev => ({ ...prev, [videoId]: !prev[videoId] }));
  };

  const handleCardClick = (videoId) => {
    navigate(`/video/${videoId}`);
  };

  const handleImageLoad = (videoId) => {
    setImagesLoaded(prev => ({ ...prev, [videoId]: true }));
  };

  const handleSearch = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleSortChange = (event) => {
    setSortCriteria(event.target.value);
  };

  const handleSortOrderChange = () => {
    setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
  };

  const handleClearFilters = () => {
    setSearchTerm('');
    setStartDate(null);
    setEndDate(null);
    setSortCriteria('date');
    setSortOrder('desc');
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return format(date, "dd/MM/yyyy 'at' HH:mm");
  };

  if (loading) {
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
        {error}
      </Typography>
    );
  }

  if (!Array.isArray(videos)) {
    return (
      <Typography color="error" align="center">
        Error: Video data is not in the expected format. Please try refreshing the page.
      </Typography>
    );
  }

  const renderVideoDetails = (videoItem) => {
    if (!videoItem || !videoItem.details) {
      console.error('Video object or details are undefined', videoItem);
      return null;
    }

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
    } = videoItem.details;

    return (
      <CardContent>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2">Video:</Typography>
        <Typography variant="body2" color="text.secondary">
          Length: {video_length || 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Total Frames: {video?.total_frames?.toLocaleString() ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Extracted Frames: {video?.extracted_frames?.toLocaleString() ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Video FPS: {video?.video_fps ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Time: {video?.video_processing_time ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Speed: {video?.video_processing_speed ?? 'N/A'}
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2">Audio:</Typography>
        <Typography variant="body2" color="text.secondary">
          Length: {audio?.audio_length ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Time: {audio?.audio_processing_time ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Speed: {audio?.audio_processing_speed ?? 'N/A'}
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2">Transcription:</Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Time: {transcription?.transcription_processing_time ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Word Count: {transcription?.word_count?.toLocaleString() ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Confidence: {transcription?.confidence ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Transcription Speed: {transcription?.transcription_speed ?? 'N/A'}
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2">Text Detection:</Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Time: {ocr?.ocr_processing_time ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Frames Processed: {ocr?.frames_processed?.toLocaleString() ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Frames with Text: {ocr?.frames_with_text?.toLocaleString() ?? 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Words Detected: {ocr?.total_words_detected?.toLocaleString() ?? 'N/A'}
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2">Total Processing:</Typography>
        <Typography variant="body2" color="text.secondary">
          Start Time: {formatDate(total_processing_start_time)}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          End Time: {formatDate(total_processing_end_time)}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Time: {total_processing_time || 'N/A'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Processing Speed: {total_processing_speed || 'N/A'}
        </Typography>
      </CardContent>
    );
  };

  return (
    <Box sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h5" gutterBottom component="div" sx={{ mb: 3 }}>
        Processed Videos History
      </Typography>
      
      <Box sx={{ backgroundColor: 'background.paper', p: 3, borderRadius: 2, boxShadow: 1 }}>
        <Grid container spacing={2} alignItems="flex-end">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search by Name or Video ID"
              variant="outlined"
              value={searchTerm}
              onChange={handleSearch}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
                endAdornment: searchTerm && (
                  <InputAdornment position="end">
                    <IconButton onClick={() => setSearchTerm('')} edge="end">
                      <ClearIcon />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <DatePicker
                    label="From Date"
                    value={startDate}
                    onChange={(newDate) => {
                      if (newDate) {
                        newDate.setHours(0, 0, 0, 0);
                      }
                      setStartDate(newDate);
                    }}
                    slotProps={{ textField: { fullWidth: true } }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <DatePicker
                    label="To Date"
                    value={endDate}
                    onChange={(newDate) => {
                      if (newDate) {
                        newDate.setHours(23, 59, 59, 999);
                      }
                      setEndDate(newDate);
                    }}
                    slotProps={{ textField: { fullWidth: true } }}
                  />
                </Grid>
              </Grid>
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} md={4}>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <FormControl fullWidth>
                  <InputLabel id="sort-by-label">Sort By</InputLabel>
                  <Select
                    labelId="sort-by-label"
                    value={sortCriteria}
                    onChange={handleSortChange}
                    label="Sort By"
                  >
                    <MenuItem value="date">Processing Date</MenuItem>
                    <MenuItem value="length">Video Length</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <Button 
                  onClick={handleSortOrderChange}
                  variant="outlined"
                  fullWidth
                >
                  {sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                </Button>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            color="secondary"
            onClick={handleClearFilters}
            startIcon={<ClearIcon />}
          >
            Clear Filters
          </Button>
        </Box>
      </Box>

      <Divider sx={{ my: 4 }} />
      
      <Grid container spacing={3}>
        {filteredVideos.map((video) => (
        <Grid item xs={12} sm={6} md={4} key={video.video_id}>
          <Card>
            <Box sx={{ position: 'relative', height: 140 }}>
              {(!firstFrames[video.video_id] || !imagesLoaded[video.video_id]) && (
                <Skeleton variant="rectangular" width="100%" height={140} />
              )}
              {firstFrames[video.video_id] && (
                <CardMedia
                  component="img"
                  height="140"
                  image={firstFrames[video.video_id]}
                  alt={`First frame of ${video.details.name || video.video_id}`}
                  onClick={() => handleCardClick(video.video_id)}
                  onLoad={() => handleImageLoad(video.video_id)}
                  onError={(e) => {
                    console.error('Error loading image for:', video.video_id);
                    e.target.src = '/path/to/fallback-image.jpg';
                  }}
                  sx={{
                    cursor: 'pointer',
                    display: imagesLoaded[video.video_id] ? 'block' : 'none',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                  }}
                />
              )}
              </Box>
              <CardContent>
                <Typography variant="h6" component="div" noWrap>
                  {video.details.name || video.video_id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Length: {video.details.video_length || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Processed: {formatDate(video.details.total_processing_end_time)}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Button
                    size="small"
                    onClick={() => handleExpandClick(video.video_id)}
                    endIcon={<ExpandMoreIcon />}
                  >
                    {expanded[video.video_id] ? 'Hide Details' : 'Show Details'}
                  </Button>
                </Box>
              </CardContent>
              <Collapse in={expanded[video.video_id]} timeout="auto" unmountOnExit>
                {renderVideoDetails(video)}
              </Collapse>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default VideoHistory;