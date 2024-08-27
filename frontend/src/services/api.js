import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

const api = {
  uploadVideo: async (file) => {
    const formData = new FormData();
    formData.append('video', file);
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getVideoStatus: async (videoId) => {
    const response = await axios.get(`${API_BASE_URL}/status/${videoId}`);
    return response.data;
  },

  getProcessedVideos: async () => {
    const response = await axios.get(`${API_BASE_URL}/processed-videos`);
    return response.data;
  },

  getVideoDetails: async (videoId) => {
    const response = await axios.get(`${API_BASE_URL}/video/${videoId}`);
    return response.data;
  },

  getVideoFrames: async (videoId) => {
    const response = await axios.get(`${API_BASE_URL}/video/${videoId}/frames`);
    return response.data;
  },

  getTranscript: async (videoId) => {
    const response = await axios.get(`${API_BASE_URL}/video/${videoId}/transcript`);
    return response.data;
  },

  downloadFile: async (videoId, fileType) => {
    const response = await fetch(`${API_BASE_URL}/video/${videoId}/download/${fileType}`, {
      method: 'GET',
      headers: {
        // Add any necessary headers here
      },
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response;
  },
};

export default api;