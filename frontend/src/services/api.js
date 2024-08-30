import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

const api = {
  uploadVideo: async (file, onProgress) => {
    const formData = new FormData();
    formData.append('video', file);

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

  getOcrResults: (videoId) => 
    axios.get(`${API_BASE_URL}/video/${videoId}/ocr/results`)
      .then(response => response.data)
      .catch(error => {
        if (error.response && error.response.status === 404) {
          throw new Error('OCR results not found');
        }
        throw error;
      }),

  getProcessedOcrResults: (videoId) => 
    axios.get(`${API_BASE_URL}/video/${videoId}/ocr/processed-ocr`)
      .then(response => response.data)
      .catch(error => {
        if (error.response && error.response.status === 404) {
          throw new Error('Processed OCR results not found');
        }
        throw error;
      }),

  getBrandsOcrResults: (videoId) => 
    axios.get(`${API_BASE_URL}/video/${videoId}/ocr/brands-ocr`)
      .then(response => response.data)
      .catch(error => {
        if (error.response && error.response.status === 404) {
          throw new Error('Brands OCR results not found');
        }
        throw error;
      }),

  getOcrWordCloud: (videoId) => 
    axios.get(`${API_BASE_URL}/video/${videoId}/ocr/wordcloud`, { responseType: 'arraybuffer' })
      .then(response => {
        const blob = new Blob([response.data], { type: 'image/jpeg' });
        return URL.createObjectURL(blob);
      })
      .catch(error => {
        if (error.response && error.response.status === 404) {
          throw new Error('Word cloud not found');
        }
        throw error;
      }),
      
  getBrandsOcrTable: (videoId) => 
    axios.get(`${API_BASE_URL}/video/${videoId}/ocr/brands-ocr-table`)
      .then(response => response.data)
      .catch(error => {
        if (error.response && error.response.status === 404) {
          throw new Error('Brands OCR table not found');
        }
        throw error;
      }),

  // New method to get processing stats
  getProcessingStats: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/processing-stats`);
      return response.data;
    } catch (error) {
      if (error.response && error.response.status === 404) {
        throw new Error('Processing stats not found');
      }
      throw error;
    }
  },
};

export default api;