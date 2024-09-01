import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes in milliseconds

const isCacheValid = (timestamp) => {
  return timestamp && Date.now() - timestamp < CACHE_DURATION;
};

const api = {
  getProcessedVideos: async () => {
    const cacheKey = 'processedVideos';
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);

    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/processed-videos`);
      const data = Array.isArray(response.data) ? response.data : [];
      localStorage.setItem(cacheKey, JSON.stringify(data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return data;
    } catch (error) {
      throw error;
    }
  },

  getVideoDetails: async (videoId) => {
    const cacheKey = `videoDetails_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);

    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getVideoFrames: async (videoId) => {
    const cacheKey = `videoFrames_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/frames`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getTranscript: async (videoId) => {
    const cacheKey = `transcript_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/transcript`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getOcrResults: async (videoId) => {
    const cacheKey = `ocrResults_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/results`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getOcrWordCloud: async (videoId) => {
    const cacheKey = `ocrWordCloud_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/wordcloud`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getBrandsOcrTable: async (videoId) => {
    const cacheKey = `brandsOcrTable_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/brands-ocr-table`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getProcessingStats: async (videoId) => {
    const cacheKey = `processingStats_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/processing-stats`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  updateVideoName: async (videoId, newName) => {
    try {
      const response = await axios.put(`${API_BASE_URL}/video/${videoId}/update-name`, { name: newName });
      // Invalidate relevant caches
      localStorage.removeItem(`videoDetails_${videoId}`);
      localStorage.removeItem(`videoDetails_${videoId}_timestamp`);
      localStorage.removeItem(`processedVideos`);
      localStorage.removeItem(`processedVideos_timestamp`);
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