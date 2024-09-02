import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Button, Box, Grid, Paper, Divider } from '@mui/material';
import VideoUploadPage from './components/VideoUploadPage';
import VideoHistory from './components/VideoHistory';
import VideoDetails from './components/VideoDetails';
import FAQ from './components/FAQ';
import Footer from './components/Footer';

function App() {
  return (
    <Router>
      <Box display="flex" flexDirection="column" minHeight="100vh">
        <AppBar position="static">
          <Toolbar>
           <Box
              component="img"
              sx={{
                height: 40,
                width: 40,
                mr: 2
              }}
              alt="Logo"
              src="/logo512.png"
            />
            <Typography variant="h6" style={{ flexGrow: 1 }}>
              Product Placement Detection
            </Typography>
            <Button color="inherit" component={Link} to="/">Home</Button>
            <Button color="inherit" component={Link} to="/upload">Upload</Button>
            <Button color="inherit" component={Link} to="/history">History</Button>
            <Button color="inherit" component={Link} to="/faq">FAQ</Button>
          </Toolbar>
        </AppBar>
        <Container component="main" sx={{ flexGrow: 1, py: 3 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<VideoUploadPage />} />
            <Route path="/history" element={<VideoHistory />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/video/:videoId" element={<VideoDetails />} />
          </Routes>
        </Container>
        <Footer />
      </Box>
    </Router>
  );
}

function Home() {
  return (
    <Box sx={{ textAlign: 'center', py: 1 }}>
      <Box
          component="img"
          sx={{
            height: 100,
            width: 100,
            mr: 2
          }}
          alt="Logo"
          src="/logo512.png"
        />
      <Typography variant="h3" gutterBottom>
        Welcome to Product Placement Detection
      </Typography>
      <Typography variant="h5" color="text.secondary" paragraph>
        Unlock the power of AI to analyze product placements in your videos
      </Typography>
      <Grid container spacing={4} justifyContent="center" sx={{ mt: 4, mb: 12 }}>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Upload Your Video
            </Typography>
            <Typography>
              Start by uploading your video file. We support various formats including MP4, AVI, and MOV.
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              AI-Powered Analysis
            </Typography>
            <Typography>
              Our advanced AI algorithms detect and track product placements throughout your video.
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Comprehensive Results
            </Typography>
            <Typography>
              Get detailed reports on product appearances, screen time, and more.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
      <Divider sx={{ my: 4 }} />
      <Typography variant="h4" gutterBottom>
        How it works
      </Typography>
      <Typography variant="body1" paragraph>
        Our product placement detection system uses a combination of computer vision and machine learning techniques to analyze videos. Here's a high-level overview of the process:
      </Typography>
      <Typography variant="body1" paragraph>
        <strong>Video Upload:</strong> Users upload their videos to our platform. 
      </Typography>
      <Typography variant="body1" paragraph>
        <strong>Video Processing:</strong> The video is processed to extract frames and audio.
      </Typography>
      <Typography variant="body1" paragraph>
        <strong>Product Detection:</strong> Our AI model detects and tracks products in the video.
      </Typography>
      <Typography variant="body1" paragraph>
        <strong>Data Analysis:</strong> We analyze the detected products and generate detailed reports.   
      </Typography>
      <Button
        variant="contained"
        color="primary"
        size="large"
        component={Link}
        to="/upload"
        sx={{ mt: 6 }}
      >
        Get Started
      </Button>
    </Box>
  );
}

export default App;