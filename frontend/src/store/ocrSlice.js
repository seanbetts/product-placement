import { createSlice, createAsyncThunk, createSelector } from '@reduxjs/toolkit';
import api from '../services/api';

// Cache duration (5 minutes)
const CACHE_DURATION = 5 * 60 * 1000;

// Helper function to check if cache is valid
const isCacheValid = (timestamp) => {
  return timestamp && Date.now() - timestamp < CACHE_DURATION;
};

// Enhanced error types
const ErrorTypes = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  API_ERROR: 'API_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
};

const initialState = {
  data: {
    wordCloud: {},
    brandTable: {},
  },
  status: {
    loading: {},
    error: {},
    errorType: {},
    fetched: {},
  },
};

export const fetchOcrData = createAsyncThunk(
  'ocr/fetchData',
  async (videoId, { getState, rejectWithValue }) => {
    const state = getState().ocr;
    const cachedWordCloud = state.data.wordCloud[videoId];
    const cachedBrandTable = state.data.brandTable[videoId];

    if (cachedWordCloud && cachedBrandTable && 
        isCacheValid(cachedWordCloud.lastFetched) && 
        isCacheValid(cachedBrandTable.lastFetched)) {
      return { 
        videoId, 
        wordCloudData: cachedWordCloud.data, 
        brandTableData: cachedBrandTable.data 
      };
    }

    try {
      const [wordCloudData, brandTableData] = await Promise.all([
        api.getOcrWordCloud(videoId),
        api.getBrandsOcrTable(videoId)
      ]);
      return { videoId, wordCloudData, brandTableData };
    } catch (error) {
      let errorType = ErrorTypes.UNKNOWN_ERROR;
      if (error.isAxiosError && !error.response) {
        errorType = ErrorTypes.NETWORK_ERROR;
      } else if (error.response) {
        errorType = ErrorTypes.API_ERROR;
      }
      return rejectWithValue({ videoId, error: error.message, errorType });
    }
  }
);

const ocrSlice = createSlice({
  name: 'ocr',
  initialState,
  reducers: {
    clearOcrData: (state, action) => {
      const videoId = action.payload;
      delete state.data.wordCloud[videoId];
      delete state.data.brandTable[videoId];
      delete state.status.loading[videoId];
      delete state.status.error[videoId];
      delete state.status.errorType[videoId];
      delete state.status.fetched[videoId];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOcrData.pending, (state, action) => {
        state.status.loading[action.meta.arg] = true;
        state.status.error[action.meta.arg] = null;
        state.status.errorType[action.meta.arg] = null;
        state.status.fetched[action.meta.arg] = false;
      })
      .addCase(fetchOcrData.fulfilled, (state, action) => {
        const { videoId, wordCloudData, brandTableData } = action.payload;
        state.status.loading[videoId] = false;
        state.status.fetched[videoId] = true;
        state.data.wordCloud[videoId] = {
          data: wordCloudData,
          lastFetched: Date.now(),
        };
        state.data.brandTable[videoId] = {
          data: brandTableData,
          lastFetched: Date.now(),
        };
      })
      .addCase(fetchOcrData.rejected, (state, action) => {
        state.status.loading[action.meta.arg] = false;
        state.status.error[action.meta.arg] = action.payload.error;
        state.status.errorType[action.meta.arg] = action.payload.errorType;
        state.status.fetched[action.meta.arg] = true;
      });
  },
});

export const { clearOcrData } = ocrSlice.actions;

export const selectOcrWordCloud = createSelector(
  [(state) => state.ocr.data.wordCloud, (_, videoId) => videoId],
  (wordCloud, videoId) => wordCloud[videoId]?.data
);

export const selectBrandsOcrTable = createSelector(
  [(state) => state.ocr.data.brandTable, (_, videoId) => videoId],
  (brandTable, videoId) => brandTable[videoId]?.data
);

export const selectOcrStatus = createSelector(
  [(state) => state.ocr.status, (_, videoId) => videoId],
  (status, videoId) => ({
    loading: status.loading[videoId] || false,
    error: status.error[videoId] || null,
    errorType: status.errorType[videoId] || null,
    fetched: status.fetched[videoId] || false,
  })
);

export default ocrSlice.reducer;