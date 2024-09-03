import React, { useEffect, useCallback, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Typography,
  Box,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress
} from '@mui/material';
import { 
  fetchOcrWordCloud, 
  fetchBrandsOcrTable, 
  selectOcrWordCloud,
  selectBrandsOcrTable,
  selectOcrStatus
} from '../../store/ocrSlice';

const capitalizeWords = (str) => {
  return str.replace(/\b\w/g, (char) => char.toUpperCase());
};

const WordCloudDisplay = React.memo(({ wordCloud }) => (
  <Box
    sx={{
      height: 400,
      width: '100%',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      bgcolor: 'white',
      border: '1px solid #ccc',
      borderRadius: '4px',
    }}
  >
    {wordCloud ? (
      <img 
        src={wordCloud} 
        alt="Word Cloud" 
        style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
      />
    ) : (
      <Typography color="text.secondary">
        No word cloud available
      </Typography>
    )}
  </Box>
));

const BrandTable = React.memo(({ brandTable }) => (
  <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
    <Table stickyHeader aria-label="brand frequency table">
      <TableHead>
        <TableRow>
          <TableCell>
            <Typography fontWeight="bold">Brand</Typography>
          </TableCell>
          <TableCell align="right">
            <Typography fontWeight="bold"># Frames</Typography>
          </TableCell>
          <TableCell align="right">
            <Typography fontWeight="bold">Time on Screen (s)</Typography>
          </TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {brandTable && Object.entries(brandTable).map(([brand, data]) => (
          <TableRow key={brand}>
            <TableCell component="th" scope="row">
              {capitalizeWords(brand)}
            </TableCell>
            <TableCell align="right">{data.frame_count}</TableCell>
            <TableCell align="right">{data.time_on_screen.toFixed(1)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
));

const TextDetectionSection = ({ videoId }) => {
  const dispatch = useDispatch();
  const wordCloud = useSelector(state => selectOcrWordCloud(state, videoId));
  const brandTable = useSelector(state => selectBrandsOcrTable(state, videoId));
  const { loading, error, errorType } = useSelector(selectOcrStatus);

  const fetchData = useCallback(async () => {
    if (!wordCloud) {
      await dispatch(fetchOcrWordCloud(videoId));
    }
    if (!brandTable) {
      await dispatch(fetchBrandsOcrTable(videoId));
    }
  }, [dispatch, videoId, wordCloud, brandTable]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const content = useMemo(() => {
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

    return (
      <Grid container spacing={2} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <WordCloudDisplay wordCloud={wordCloud} />
        </Grid>
        <Grid item xs={12} md={6}>
          <BrandTable brandTable={brandTable} />
        </Grid>
      </Grid>
    );
  }, [loading, error, errorType, wordCloud, brandTable]);

  return (
    <Box sx={{ mt: 4 }}>
      {content}
    </Box>
  );
};

export default React.memo(TextDetectionSection);