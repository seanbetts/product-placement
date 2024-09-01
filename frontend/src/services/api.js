import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes in milliseconds

const isCacheValid = (timestamp) => {
  return timestamp && Date.now() - timestamp < CACHE_DURATION;
};

const api = {
  getProcessedVideos: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/processed-videos`);
      return Array.isArray(response.data) ? response.data : [];
    } catch (error) {
      throw error;
    }
  },

  getVideoDetails: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getVideoFrames: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/frames`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getTranscript: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/transcript`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getOcrResults: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/results`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getOcrWordCloud: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/wordcloud`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getBrandsOcrTable: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/brands-ocr-table`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getProcessingStats: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/processing-stats`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  updateVideoName: async (videoId, newName) => {
    try {
      const response = await axios.put(`${API_BASE_URL}/video/${videoId}/update-name`, { name: newName });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  uploadVideo: async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        },
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getVideoStatus: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status/${videoId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  downloadFile: async (videoId, fileType) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/download/${fileType}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};

export default api;