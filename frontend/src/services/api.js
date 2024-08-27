import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('video', file);

  const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

const getVideoStatus = async (videoId) => {
  const response = await axios.get(`${API_BASE_URL}/status/${videoId}`);
  return response.data;
};

const getProcessedVideos = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/processed-videos`);
    console.log('API response:', response.data); // Log the API response
    return response.data;
  } catch (error) {
    console.error('Error in getProcessedVideos:', error);
    throw error;
  }
};

export default {
  uploadVideo,
  getVideoStatus,
  getProcessedVideos
};