import React from 'react';
import { Container, Typography, Button } from '@mui/material';
import Header from './components/Header';

function App() {
  return (
    <div className="App">
      <Header />
      <Container>
        <Typography variant="h5" sx={{ margin: '20px 0' }}>
          Welcome to Product Placement Detection
        </Typography>
        <Button variant="contained" color="primary">
          Upload Video (Coming Soon)
        </Button>
        <Typography variant="body1" sx={{ margin: '20px 0' }}>
          Status: Frontend is running successfully!
        </Typography>
      </Container>
    </div>
  );
}

export default App;