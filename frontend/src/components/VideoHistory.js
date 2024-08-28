import React, { useState, useEffect, useCallback } from 'react';
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
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ClearIcon from '@mui/icons-material/Clear';
import { useNavigate } from 'react-router-dom';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { format } from 'date-fns';
import enGB from 'date-fns/locale/en-GB';
import api from '../services/api';

const VideoHistory = () => {
  const [videos, setVideos] = useState([]);
  const [filteredVideos, setFilteredVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [imagesLoaded, setImagesLoaded] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [sortCriteria, setSortCriteria] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');
  const navigate = useNavigate();

  const filterAndSortVideos = useCallback(() => {
    let result = videos.filter(video => 
      video.details.video_id.toLowerCase().includes(searchTerm.toLowerCase()) &&
      (!startDate || new Date(video.details.total_processing_end_time) >= startDate) &&
      (!endDate || new Date(video.details.total_processing_end_time) <= endDate)
    );

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

    setFilteredVideos(result);
  }, [videos, searchTerm, startDate, endDate, sortCriteria, sortOrder]);

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const data = await api.getProcessedVideos();
        console.log('Fetched videos:', data);

        // Sort videos by processing date and time in descending order
        const sortedVideos = data.sort((a, b) => {
          const dateA = new Date(a.details.total_processing_end_time).getTime();
          const dateB = new Date(b.details.total_processing_end_time).getTime();
          return dateB - dateA;
        });

        setVideos(sortedVideos);
        setFilteredVideos(sortedVideos);
        setExpanded(sortedVideos.reduce((acc, video) => ({ ...acc, [video.video_id]: false }), {}));
        setImagesLoaded(sortedVideos.reduce((acc, video) => ({ ...acc, [video.video_id]: false }), {}));
        setLoading(false);
      } catch (err) {
        console.error('Error fetching processed videos:', err);
        setError('Failed to load video history. Please try again later.');
        setLoading(false);
      }
    };

    fetchVideos();
  }, []);

  useEffect(() => {
    filterAndSortVideos();
  }, [filterAndSortVideos]);

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
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
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

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom component="div" sx={{ mb: 3 }}>
        Processed Videos History
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={3}>
            <TextField
              fullWidth
              label="Search by Video ID"
              variant="outlined"
              value={searchTerm}
              onChange={handleSearch}
              InputProps={{
                endAdornment: (
                  <IconButton onClick={() => setSearchTerm('')}>
                    <ClearIcon />
                  </IconButton>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} sm={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns} adapterLocale={enGB}>
              <DatePicker
                label="Filter From Date"
                value={startDate}
                onChange={setStartDate}
                renderInput={(params) => <TextField {...params} fullWidth />}
                inputFormat="dd/MM/yyyy"
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} sm={3}>
            <LocalizationProvider dateAdapter={AdapterDateFns} adapterLocale={enGB}>
              <DatePicker
                label="Filter To Date"
                value={endDate}
                onChange={setEndDate}
                renderInput={(params) => <TextField {...params} fullWidth />}
                inputFormat="dd/MM/yyyy"
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} sm={3}>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleClearFilters}
              startIcon={<ClearIcon />}
              fullWidth
            >
              Clear Filters
            </Button>
          </Grid>
        </Grid>
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Grid container spacing={2} alignItems="flex-end">
          <Grid item xs={12} sm={4}>
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
          <Grid item xs={12} sm={4}>
            <Button 
              onClick={handleSortOrderChange}
              variant="outlined"
              fullWidth
            >
              {sortOrder === 'asc' ? 'Ascending' : 'Descending'}
            </Button>
          </Grid>
        </Grid>
      </Box>

      <Grid container spacing={3}>
        {filteredVideos.map((video) => (
          <Grid item xs={12} sm={6} md={4} key={video.video_id}>
            <Card>
              <Box sx={{ position: 'relative' }}>
                {!imagesLoaded[video.video_id] && (
                  <Skeleton
                    variant="rectangular"
                    width="100%"
                    height={140}
                    animation="wave"
                  />
                )}
                <CardMedia
                  component="img"
                  height="140"
                  image={`${process.env.REACT_APP_API_URL}/video-frame/${video.details.video_id}`}
                  alt={`First frame of video ${video.details.video_id}`}
                  onClick={() => handleCardClick(video.details.video_id)}
                  sx={{
                    cursor: 'pointer',
                    display: imagesLoaded[video.video_id] ? 'block' : 'none'
                  }}
                  onLoad={() => handleImageLoad(video.video_id)}
                />
              </Box>
              <CardContent>
                <Typography variant="h6" component="div" noWrap>
                  {video.details.video_id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Length: {video.details.video_length || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Processed: {formatDate(new Date(video.details.total_processing_end_time))}
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
                <CardContent>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Video:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Frames: {video.details.video.total_frames || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Extracted Frames: {video.details.video.extracted_frames || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Video FPS: {video.details.video.video_fps || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time: {video.details.video.video_processing_time || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Speed: {video.details.video.video_processing_speed || 'N/A'}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Audio:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Length: {video.details.audio.audio_length || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time: {video.details.audio.audio_processing_time || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Speed: {video.details.audio.audio_processing_speed || 'N/A'}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Transcription:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time: {video.details.transcription.transcription_processing_time || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Word Count {video.details.transcription.word_count || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence {video.details.transcription.confidence || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Transcription Speed {video.details.transcription.transcription_speed || 'N/A'}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2">Total Processing:</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Start Time: {new Date(video.details.total_processing_start_time).toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    End Time: {new Date(video.details.total_processing_end_time).toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time: {video.details.total_processing_time || 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Processing Speed: {video.details.total_processing_speed || 'N/A'}
                  </Typography>
                </CardContent>
              </Collapse>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default VideoHistory;