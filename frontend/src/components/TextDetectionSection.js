import React, { useMemo, useRef, useEffect } from 'react';
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
  Alert 
} from '@mui/material';
import * as d3 from 'd3';
import cloud from 'd3-cloud';

const WordCloud = ({ words }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (words.length === 0) return;

    const width = 800;
    const height = 400;
    const padding = 2;

    const fontSize = d3.scaleLinear()
      .domain([d3.min(words, d => d.value), d3.max(words, d => d.value)])
      .range([15, 80]);

    const layout = cloud()
      .size([width - padding * 2, height - padding * 2])
      .words(words.map(d => ({ ...d, size: fontSize(d.value) })))
      .padding(3)
      .rotate(0)
      .font("Arial")
      .fontSize(d => d.size)
      .random(() => 0.5)
      .spiral("archimedean")
      .on("end", draw);

    layout.start();

    function draw(words) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, width, height);

      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      const color = d3.scaleOrdinal(d3.schemeCategory10);

      words.forEach(word => {
        const x = word.x + width / 2;
        const y = word.y + height / 2;
        ctx.font = `${Math.round(word.size)}px Arial`;
        ctx.fillStyle = color(word.text);
        ctx.fillText(word.text, x, y);
      });
    }
  }, [words]);

  return <canvas ref={canvasRef} width={800} height={400}></canvas>;
};

const TextDetectionSection = ({ ocrError, wordCloudData, videoFps }) => {
  const wordTableData = useMemo(() => {
    if (!wordCloudData || wordCloudData.length === 0) return [];

    return wordCloudData
      .map(({ text, value }) => ({
        word: text,
        frequency: value,
        timeOnScreen: (value / videoFps).toFixed(1)
      }))
      .sort((a, b) => b.frequency - a.frequency);
  }, [wordCloudData, videoFps]);

  if (ocrError) {
    return <Alert severity="info">{ocrError}</Alert>;
  }

  if (wordCloudData.length === 0) {
    return <Alert severity="info">No Text Detection data available for this video.</Alert>;
  }

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>Text Detection</Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Box 
            className="word-cloud-container" 
            sx={{ 
              height: 400, 
              width: '100%', 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              bgcolor: 'white' 
            }}
          >
            <WordCloud words={wordCloudData} />
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <TableContainer component={Paper} sx={{ maxHeight: 400, overflow: 'auto' }}>
            <Table stickyHeader aria-label="word frequency table">
              <TableHead>
                <TableRow>
                  <TableCell>Word</TableCell>
                  <TableCell align="right"># Frames</TableCell>
                  <TableCell align="right">Time on Screen (s)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {wordTableData.map((row) => (
                  <TableRow key={row.word}>
                    <TableCell component="th" scope="row">
                      {row.word}
                    </TableCell>
                    <TableCell align="right">{row.frequency}</TableCell>
                    <TableCell align="right">{row.timeOnScreen}</TableCell>
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