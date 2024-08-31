import axios from 'axios';
import { 
  setVideos, 
  setVideoDetails, 
  setLoading as setVideoLoading, 
  setError as setVideoError 
} from '../store/videoSlice';
import { 
  setTranscript, 
  setLoading as setTranscriptLoading, 
  setError as setTranscriptError 
} from '../store/transcriptSlice';
import { 
  setOcrResults, 
  setWordCloud, 
  setBrandTable, 
  setProcessingStats, 
  setLoading as setOcrLoading, 
  setError as setOcrError 
} from '../store/ocrSlice';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes in milliseconds

const isCacheValid = (timestamp) => {
  return timestamp && Date.now() - timestamp < CACHE_DURATION;
};

const api = {
  getProcessedVideos: async (dispatch) => {
    try {
      dispatch(setVideoLoading(true));
      const response = await axios.get(`${API_BASE_URL}/processed-videos`);
      const videos = response.data;
      
      // Ensure videos is always an array
      const videoArray = Array.isArray(videos) ? videos : [];
      
      dispatch(setVideos({ videos: videoArray, lastFetched: Date.now() }));
      dispatch(setVideoLoading(false));
      return videoArray;
    } catch (error) {
      dispatch(setVideoError(error.message));
      dispatch(setVideoLoading(false));
      throw error;
    }
  },

  getVideoDetails: async (videoId, dispatch, getState) => {
    const state = getState();
    const videoDetails = state.videos.details[videoId];

    if (videoDetails && isCacheValid(videoDetails.lastFetched)) {
      return videoDetails;
    }

    try {
      dispatch(setVideoLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}`);
      const details = response.data;
      dispatch(setVideoDetails({ id: videoId, details: { ...details, lastFetched: Date.now() } }));
      dispatch(setVideoLoading(false));
      return details;
    } catch (error) {
      dispatch(setVideoError(error.message));
      dispatch(setVideoLoading(false));
      throw error;
    }
  },

  getTranscript: async (videoId, dispatch, getState) => {
    const state = getState();
    const transcript = state.transcripts.data[videoId];

    if (transcript && isCacheValid(transcript.lastFetched)) {
      return transcript.data;
    }

    try {
      dispatch(setTranscriptLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/transcript`);
      const transcriptData = response.data;
      dispatch(setTranscript({ id: videoId, transcript: { data: transcriptData, lastFetched: Date.now() } }));
      dispatch(setTranscriptLoading(false));
      return transcriptData;
    } catch (error) {
      dispatch(setTranscriptError(error.message));
      dispatch(setTranscriptLoading(false));
      throw error;
    }
  },

  getOcrResults: async (videoId, dispatch, getState) => {
    const state = getState();
    const ocrResults = state.ocr.results[videoId];

    if (ocrResults && isCacheValid(ocrResults.lastFetched)) {
      return ocrResults.data;
    }

    try {
      dispatch(setOcrLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/results`);
      const results = response.data;
      dispatch(setOcrResults({ id: videoId, results: { data: results, lastFetched: Date.now() } }));
      dispatch(setOcrLoading(false));
      return results;
    } catch (error) {
      dispatch(setOcrError(error.message));
      dispatch(setOcrLoading(false));
      throw error;
    }
  },

  getOcrWordCloud: async (videoId, dispatch, getState) => {
    const state = getState();
    const wordCloud = state.ocr.wordCloud[videoId];

    if (wordCloud && isCacheValid(wordCloud.lastFetched)) {
      return wordCloud.url;
    }

    try {
      dispatch(setOcrLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/wordcloud`, { responseType: 'arraybuffer' });
      const blob = new Blob([response.data], { type: 'image/jpeg' });
      const wordCloudUrl = URL.createObjectURL(blob);
      dispatch(setWordCloud({ id: videoId, wordCloud: { url: wordCloudUrl, lastFetched: Date.now() } }));
      dispatch(setOcrLoading(false));
      return wordCloudUrl;
    } catch (error) {
      dispatch(setOcrLoading(false));
      if (error.response && error.response.status === 404) {
        const errorMessage = 'Word cloud not found';
        dispatch(setOcrError(errorMessage));
        throw new Error(errorMessage);
      }
      dispatch(setOcrError(error.message));
      throw error;
    }
  },

  getBrandsOcrTable: async (videoId, dispatch, getState) => {
    const state = getState();
    const brandTable = state.ocr.brandTable[videoId];

    if (brandTable && isCacheValid(brandTable.lastFetched)) {
      return brandTable.data;
    }

    try {
      dispatch(setOcrLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/ocr/brands-ocr-table`);
      const tableData = response.data;
      dispatch(setBrandTable({ id: videoId, brandTable: { data: tableData, lastFetched: Date.now() } }));
      dispatch(setOcrLoading(false));
      return tableData;
    } catch (error) {
      dispatch(setOcrLoading(false));
      if (error.response && error.response.status === 404) {
        const errorMessage = 'Brands OCR table not found';
        dispatch(setOcrError(errorMessage));
        throw new Error(errorMessage);
      }
      dispatch(setOcrError(error.message));
      throw error;
    }
  },

  getProcessingStats: async (videoId, dispatch, getState) => {
    const state = getState();
    const stats = state.ocr.processingStats[videoId];

    if (stats && isCacheValid(stats.lastFetched)) {
      return stats.data;
    }

    try {
      dispatch(setOcrLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/processing-stats`);
      const statsData = response.data;
      dispatch(setProcessingStats({ id: videoId, stats: { data: statsData, lastFetched: Date.now() } }));
      dispatch(setOcrLoading(false));
      return statsData;
    } catch (error) {
      dispatch(setOcrLoading(false));
      if (error.response && error.response.status === 404) {
        const errorMessage = 'Processing stats not found';
        dispatch(setOcrError(errorMessage));
        throw new Error(errorMessage);
      }
      dispatch(setOcrError(error.message));
      throw error;
    }
  },

  updateVideoName: async (videoId, name, dispatch) => {
    try {
      dispatch(setVideoLoading(true));
      const response = await axios.post(`${API_BASE_URL}/video/${videoId}/update-name`, null, {
        params: { name }
      });
      dispatch(setVideoDetails({ id: videoId, details: { ...response.data, name } }));
      dispatch(setVideoLoading(false));
      return { success: true, data: response.data };
    } catch (error) {
      dispatch(setVideoLoading(false));
      console.error('Error updating video name:', error);
      let errorMessage;
      if (error.response) {
        errorMessage = error.response.data || error.response.statusText;
        dispatch(setVideoError(errorMessage));
        return {
          success: false,
          error: errorMessage,
          status: error.response.status
        };
      } else if (error.request) {
        errorMessage = 'No response received from server';
      } else {
        errorMessage = 'Error setting up the request';
      }
      dispatch(setVideoError(errorMessage));
      return { success: false, error: errorMessage };
    }
  },
};

export default api;