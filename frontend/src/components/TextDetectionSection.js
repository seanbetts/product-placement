import React, { useEffect } from 'react';
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
import { fetchOcrWordCloud, fetchBrandsOcrTable } from '../store/ocrSlice';
import { createSelector } from '@reduxjs/toolkit';

// Create a memoized selector
const selectOcrData = createSelector(
  [
    (state) => state.ocr.data.wordCloud,
    (state) => state.ocr.data.brandTable,
    (state) => state.ocr.status.loading,
    (state) => state.ocr.status.error,
    (_, videoId) => videoId
  ],
  (wordCloud, brandTable, loading, error, videoId) => ({
    wordCloud: wordCloud[videoId],
    brandTable: brandTable[videoId],
    loading,
    error
  })
);

const capitalizeWords = (str) => {
  return str.replace(/\b\w/g, (char) => char.toUpperCase());
};

const TextDetectionSection = ({ videoId }) => {
  const dispatch = useDispatch();

  // Use the memoized selector
  const { wordCloud, brandTable, loading, error } = useSelector(state => selectOcrData(state, videoId));

  useEffect(() => {
    if (!wordCloud) {
      dispatch(fetchOcrWordCloud(videoId));
    }
    if (!brandTable) {
      dispatch(fetchBrandsOcrTable(videoId));
    }
  }, [dispatch, videoId, wordCloud, brandTable]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Typography color="error">{error}</Typography>;
  }

  return (
    <Box sx={{ mt: 4 }}>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
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
        </Grid>

        <Grid item xs={12} md={6}>
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
                {brandTable && brandTable.data && Object.entries(brandTable.data).map(([brand, data]) => (
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
        </Grid>
      </Grid>
    </Box>
  );
};

export default TextDetectionSection;