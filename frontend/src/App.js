import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Container, Button, Box } from '@mui/material';
import VideoUploadPage from './components/VideoUploadPage';
import VideoHistory from './components/VideoHistory';
import VideoDetails from './components/VideoDetails';
import Footer from './components/Footer';

function App() {
  return (
    <Router>
      <Box display="flex" flexDirection="column" minHeight="100vh">
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" style={{ flexGrow: 1 }}>
              Product Placement Detection
            </Typography>
            <Button color="inherit" component={Link} to="/">Home</Button>
            <Button color="inherit" component={Link} to="/upload">Upload</Button>
            <Button color="inherit" component={Link} to="/history">History</Button>
          </Toolbar>
        </AppBar>
        <Container component="main" sx={{ flexGrow: 1, py: 3 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<VideoUploadPage />} />
            <Route path="/history" element={<VideoHistory />} />
            <Route path="/video/:videoId" element={<VideoDetails />} />
          </Routes>
        </Container>
        <Footer />
      </Box>
    </Router>
  );
}

function Home() {
  return <h2>Welcome to Product Placement Detection</h2>;
}

export default App;