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
  Alert,
  CircularProgress
} from '@mui/material';
import { fetchOcrWordCloud, fetchBrandsOcrTable } from '../store/ocrSlice';

const capitalizeWords = (str) => {
  return str.replace(/\b\w/g, (char) => char.toUpperCase());
};

const TextDetectionSection = ({ videoId }) => {
  const dispatch = useDispatch();

  const { wordCloud, brandTable, loading, error } = useSelector(state => ({
    wordCloud: state.ocr.data.wordCloud[videoId],
    brandTable: state.ocr.data.brandTable[videoId],
    loading: state.ocr.status.loading,
    error: state.ocr.status.error
  }));

  useEffect(() => {
    if (!wordCloud) {
      dispatch(fetchOcrWordCloud(videoId));
    }
    if (!brandTable) {
      dispatch(fetchBrandsOcrTable(videoId));
    }
  }, [dispatch, videoId, wordCloud, brandTable]);

  if (loading) {
    return <CircularProgress />;
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!wordCloud || !brandTable) {
    return <Typography>Loading text detection data...</Typography>;
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
              bgcolor: 'white'
            }}
          >
            {wordCloud && wordCloud.url && (
              <img 
                src={wordCloud.url} 
                alt="Word Cloud" 
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
              />
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