import React, { useEffect, useMemo } from 'react';
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
  fetchOcrData, 
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

const BrandTable = React.memo(({ brandTable }) => {
  const sortedBrands = useMemo(() => {
    return Object.entries(brandTable || {})
      .sort((a, b) => b[1].time_on_screen - a[1].time_on_screen);
  }, [brandTable]);

  return (
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
          {sortedBrands.map(([brand, data]) => (
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
  );
});

const TextDetectionSection = ({ videoId }) => {
  const dispatch = useDispatch();
  const wordCloud = useSelector(state => selectOcrWordCloud(state, videoId));
  const brandTable = useSelector(state => selectBrandsOcrTable(state, videoId));
  const { loading, error, fetched } = useSelector(state => selectOcrStatus(state, videoId));

  useEffect(() => {
    if (!fetched && !loading && !error) {
      dispatch(fetchOcrData(videoId));
    }
  }, [dispatch, videoId, fetched, loading, error]);

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
      return <Typography color="error">{error}</Typography>;
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
  }, [loading, error, wordCloud, brandTable]);

  return (
    <Box sx={{ mt: 4 }}>
      {content}
    </Box>
  );
};

export default React.memo(TextDetectionSection);