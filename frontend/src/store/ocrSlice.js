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

// Async thunks
export const fetchOcrWordCloud = createAsyncThunk(
  'ocr/fetchWordCloud',
  async (videoId, { getState, rejectWithValue }) => {
    const state = getState().ocr;
    const cachedData = state.data.wordCloud[videoId];

    if (cachedData && isCacheValid(cachedData.lastFetched)) {
      return { videoId, wordCloudData: cachedData.data };
    }

    try {
      const wordCloudData = await api.getOcrWordCloud(videoId);
      return { videoId, wordCloudData };
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

export const fetchBrandsOcrTable = createAsyncThunk(
  'ocr/fetchBrandsTable',
  async (videoId, { getState, rejectWithValue }) => {
    const state = getState().ocr;
    const cachedData = state.data.brandTable[videoId];

    if (cachedData && isCacheValid(cachedData.lastFetched)) {
      return { videoId, brandTableData: cachedData.data };
    }

    try {
      const brandTableData = await api.getBrandsOcrTable(videoId);
      return { videoId, brandTableData };
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

const initialState = {
  data: {
    wordCloud: {},
    brandTable: {},
  },
  status: {
    loading: false,
    error: null,
    errorType: null,
  },
};

const ocrSlice = createSlice({
  name: 'ocr',
  initialState,
  reducers: {
    clearOcrData: (state, action) => {
      const videoId = action.payload;
      delete state.data.wordCloud[videoId];
      delete state.data.brandTable[videoId];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOcrWordCloud.pending, (state) => {
        state.status.loading = true;
        state.status.error = null;
        state.status.errorType = null;
      })
      .addCase(fetchOcrWordCloud.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data.wordCloud[action.payload.videoId] = {
          data: action.payload.wordCloudData,
          lastFetched: Date.now(),
        };
      })
      .addCase(fetchOcrWordCloud.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload.error;
        state.status.errorType = action.payload.errorType;
      })
      .addCase(fetchBrandsOcrTable.pending, (state) => {
        state.status.loading = true;
        state.status.error = null;
        state.status.errorType = null;
      })
      .addCase(fetchBrandsOcrTable.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data.brandTable[action.payload.videoId] = {
          data: action.payload.brandTableData,
          lastFetched: Date.now(),
        };
      })
      .addCase(fetchBrandsOcrTable.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload.error;
        state.status.errorType = action.payload.errorType;
      });
  },
});

export const { clearOcrData } = ocrSlice.actions;

// Memoized selectors
export const selectOcrWordCloud = createSelector(
  [(state) => state.ocr.data.wordCloud, (_, videoId) => videoId],
  (wordCloud, videoId) => wordCloud[videoId]?.data
);

export const selectBrandsOcrTable = createSelector(
  [(state) => state.ocr.data.brandTable, (_, videoId) => videoId],
  (brandTable, videoId) => brandTable[videoId]?.data
);

export const selectOcrStatus = createSelector(
  [(state) => state.ocr.status],
  (status) => status
);

export default ocrSlice.reducer;