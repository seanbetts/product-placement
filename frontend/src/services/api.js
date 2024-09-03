import axios from 'axios'
import { handleApiError } from '../utils/errorHandling';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes in milliseconds

const isCacheValid = (timestamp) => {
  return timestamp && Date.now() - timestamp < CACHE_DURATION;
};

const invalidateCache = (keys) => {
  keys.forEach(key => {
    localStorage.removeItem(key);
    localStorage.removeItem(`${key}_timestamp`);
  });
};

const api = {
  uploadVideo: async (file, onProgress, cancelSignal) => {
    const chunkSize = 5 * 1024 * 1024; // 5MB chunks
    const totalChunks = Math.ceil(file.size / chunkSize);
    let videoId = null;

    for (let chunkNumber = 1; chunkNumber <= totalChunks; chunkNumber++) {
        if (cancelSignal.isCancelled) {
            if (videoId) {
                await axios.post(`${API_BASE_URL}/video/cancel-upload/${videoId}`);
            }
            throw new Error('Upload cancelled');
        }

        const start = (chunkNumber - 1) * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const chunk = file.slice(start, end);
        const formData = new FormData();
        formData.append('file', chunk, file.name);
        formData.append('chunk_number', chunkNumber);
        formData.append('total_chunks', totalChunks);
        if (videoId) formData.append('video_id', videoId);

        try {
            const response = await axios.post(`${API_BASE_URL}/video/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round(
                        ((chunkNumber - 1) / totalChunks + progressEvent.loaded / (chunkSize * totalChunks)) * 100
                    );
                    onProgress(percentCompleted);
                },
            });

            videoId = response.data.video_id;
            if (response.data.status === 'processing') {
                invalidateCache(['processedVideos']);
                return response.data;
            }
        } catch (error) {
            if (videoId) {
                await axios.post(`${API_BASE_URL}/video/cancel-upload/${videoId}`);
            }
            throw error;
        }
    }
    throw new Error('Upload failed to complete');
  },

  cancelUpload: async (videoId) => {
    await axios.post(`${API_BASE_URL}/video/cancel-upload/${videoId}`);
    invalidateCache(['processedVideos']);
  },

  getProcessedVideos: async () => {
    const cacheKey = 'processedVideos';
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);

    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/video/processed-videos`);
      const data = Array.isArray(response.data) ? response.data : [];
      localStorage.setItem(cacheKey, JSON.stringify(data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  getVideoStatus: async (videoId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/${videoId}/video/status`);
      invalidateCache([`processingStats_${videoId}`, `videoDetails_${videoId}`]);
      return response.data;
    } catch (error) {
      throw handleApiError(error);
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
      const response = await axios.get(`${API_BASE_URL}/${videoId}/video/processing-stats`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  updateVideoName: async (videoId, newName) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/${videoId}/video/update-name`, { name: newName });
      invalidateCache([
        `videoDetails_${videoId}`,
        'processedVideos',
        `processingStats_${videoId}`
      ]);
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  getFirstVideoFrame: async (videoId) => {
    const cacheKey = `firstVideoFrame_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/${videoId}/images/first-frame`, {
        responseType: 'arraybuffer'
      });
      const base64 = btoa(
        new Uint8Array(response.data).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      );
      const imageData = `data:image/jpeg;base64,${base64}`;
      localStorage.setItem(cacheKey, JSON.stringify(imageData));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return imageData;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  getAllVideoFrames: async (videoId) => {
    const cacheKey = `videoFrames_${videoId}`;
    const cachedData = localStorage.getItem(cacheKey);
    const cachedTimestamp = localStorage.getItem(`${cacheKey}_timestamp`);
  
    if (cachedData && isCacheValid(parseInt(cachedTimestamp))) {
      return JSON.parse(cachedData);
    }
  
    try {
      const response = await axios.get(`${API_BASE_URL}/${videoId}/images/all-frames`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw handleApiError(error);
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
      const response = await axios.get(`${API_BASE_URL}/${videoId}/transcript`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  downloadFile: async (videoId, fileType) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/${videoId}/files/download/${fileType}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      throw handleApiError(error);
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
      const response = await axios.get(`${API_BASE_URL}/${videoId}/ocr/wordcloud`, {
        responseType: 'arraybuffer'
      });
      
      const base64 = btoa(
        new Uint8Array(response.data).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      );
      const imageData = `data:image/jpeg;base64,${base64}`;
      
      localStorage.setItem(cacheKey, JSON.stringify(imageData));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      
      return imageData;
    } catch (error) {
      throw handleApiError(error);
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
      const response = await axios.get(`${API_BASE_URL}/${videoId}/ocr/brands-ocr-table`);
      localStorage.setItem(cacheKey, JSON.stringify(response.data));
      localStorage.setItem(`${cacheKey}_timestamp`, Date.now().toString());
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  },

  // New function to invalidate all caches for a specific video
  invalidateVideoCache: (videoId) => {
    invalidateCache([
      `videoDetails_${videoId}`,
      `processingStats_${videoId}`,
      `firstVideoFrame_${videoId}`,
      `videoFrames_${videoId}`,
      `transcript_${videoId}`,
      `ocrWordCloud_${videoId}`,
      `brandsOcrTable_${videoId}`,
      'processedVideos' // This affects the list of all videos
    ]);
  },

  // New function to invalidate all caches
  invalidateAllCaches: () => {
    localStorage.clear();
  }
};

export default api;