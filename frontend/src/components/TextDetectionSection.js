import React, { useState, useEffect } from 'react';
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
import api from '../services/api';

const TextDetectionSection = ({ videoId }) => {
  const [wordCloudUrl, setWordCloudUrl] = useState(null);
  const [brandTable, setBrandTable] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch word cloud
        const wordCloudUrl = await api.getOcrWordCloud(videoId);
        setWordCloudUrl(wordCloudUrl);

        // Fetch brand table data
        const brandTableData = await api.getBrandsOcrTable(videoId);
        setBrandTable(brandTableData);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message || 'Failed to load text detection results. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    if (videoId) {
      fetchData();
    }
  }, [videoId]);

  useEffect(() => {
    // Cleanup function to revoke the blob URL when component unmounts or URL changes
    return () => {
      if (wordCloudUrl) {
        URL.revokeObjectURL(wordCloudUrl);
      }
    };
  }, [wordCloudUrl]);

  if (loading) {
    return <CircularProgress />;
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Box sx={{ mt: 4 }}>

      {/* Word Cloud */}
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
            {wordCloudUrl && (
              <img 
                src={wordCloudUrl} 
                alt="Word Cloud" 
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
              />
            )}
          </Box>
        </Grid>

        {/* Brand Table */}
        <Grid item xs={12} md={6}>
          <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
            <Table stickyHeader aria-label="brand frequency table">
              <TableHead>
                <TableRow>
                  <TableCell>Brand</TableCell>
                  <TableCell align="right"># Frames</TableCell>
                  <TableCell align="right">Time on Screen (s)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {brandTable && Object.entries(brandTable).map(([brand, data]) => (
                  <TableRow key={brand}>
                    <TableCell component="th" scope="row">
                      {brand.charAt(0).toUpperCase() + brand.slice(1)}
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
