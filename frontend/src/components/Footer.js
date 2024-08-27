import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const Footer = () => {
  return (
    <Box sx={{ bgcolor: 'background.paper', p: 6 }} component="footer">
      <Typography variant="body2" color="text.secondary" align="center">
        {'Copyright © '}
        <Link color="inherit" href="https://your-website.com/">
          Product Placement Detection
        </Link>{' '}
        {new Date().getFullYear()}
        {'.'}
      </Typography>
    </Box>
  );
};

export default Footer;