import axios from 'axios';
import { 
  setVideos,
  fetchVideoDetails,
  updateVideoName,
  setSnackbar,
  setVideoFrames,
  setLoading,
  setError
} from '../store/videoSlice';
import { 
  setTranscript, 
  setLoading as setTranscriptLoading, 
  setError as setTranscriptError 
} from '../store/transcriptSlice';
import { 
  setOcrResults, 
  setProcessingStats, 
  setLoading as setOcrLoading, 
  setError as setOcrError,
  fetchOcrWordCloud,
  fetchBrandsOcrTable
} from '../store/ocrSlice';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes in milliseconds

const isCacheValid = (timestamp) => {
  return timestamp && Date.now() - timestamp < CACHE_DURATION;
};

const api = {
  getProcessedVideos: async (dispatch) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/processed-videos`);
      const videos = response.data;
      
      // Ensure videos is always an array
      const videoArray = Array.isArray(videos) ? videos : [];
      
      dispatch(setVideos({ videos: videoArray, lastFetched: Date.now() }));
      return videoArray;
    } catch (error) {
      dispatch(setSnackbar({ open: true, message: 'Error fetching processed videos', severity: 'error' }));
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
      const action = await dispatch(fetchVideoDetails(videoId));
      return action.payload;
    } catch (error) {
      dispatch(setSnackbar({ open: true, message: 'Error fetching video details', severity: 'error' }));
      throw error;
    }
  },

  getVideoFrames: async (videoId, dispatch, getState) => {
    const state = getState();
    const frames = state.videos.frames[videoId];

    if (frames && isCacheValid(frames.lastFetched)) {
      return frames.data;
    }

    try {
      dispatch(setLoading(true));
      const response = await axios.get(`${API_BASE_URL}/video/${videoId}/frames`);
      const framesData = response.data;
      dispatch(setVideoFrames({ id: videoId, frames: { data: framesData, lastFetched: Date.now() } }));
      dispatch(setLoading(false));
      return framesData;
    } catch (error) {
      dispatch(setError(error.message));
      dispatch(setLoading(false));
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
      const action = await dispatch(fetchOcrWordCloud(videoId));
      return action.payload.wordCloudUrl;
    } catch (error) {
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
      const action = await dispatch(fetchBrandsOcrTable(videoId));
      return action.payload.brandTableData;
    } catch (error) {
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
      const action = await dispatch(updateVideoName({ videoId, newName: name }));
      if (action.payload.success) {
        dispatch(setSnackbar({ open: true, message: 'Video name updated successfully', severity: 'success' }));
      } else {
        throw new Error(action.payload.error);
      }
      return action.payload;
    } catch (error) {
      console.error('Error updating video name:', error);
      dispatch(setSnackbar({ open: true, message: `Error updating video name: ${error.message}`, severity: 'error' }));
      return { success: false, error: error.message };
    }
  },
};

export default api;