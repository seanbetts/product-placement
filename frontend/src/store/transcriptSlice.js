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

export const fetchTranscript = createAsyncThunk(
  'transcripts/fetchTranscript',
  async (videoId, { getState, rejectWithValue }) => {
    const state = getState().transcripts;
    const cachedTranscript = state.data[videoId];

    if (cachedTranscript && isCacheValid(cachedTranscript.lastFetched)) {
      return { videoId, transcript: cachedTranscript.data };
    }

    try {
      const transcript = await api.getTranscript(videoId);
      return { videoId, transcript };
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

const transcriptSlice = createSlice({
  name: 'transcripts',
  initialState: {
    data: {},
    status: {
      loading: false,
      error: null,
      errorType: null,
    },
  },
  reducers: {
    clearTranscript: (state, action) => {
      delete state.data[action.payload];
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTranscript.pending, (state) => {
        state.status.loading = true;
        state.status.error = null;
        state.status.errorType = null;
      })
      .addCase(fetchTranscript.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data[action.payload.videoId] = {
          data: action.payload.transcript,
          lastFetched: Date.now()
        };
      })
      .addCase(fetchTranscript.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload.error;
        state.status.errorType = action.payload.errorType;
      });
  },
});

export const { clearTranscript } = transcriptSlice.actions;

// Memoized selectors
export const selectTranscript = createSelector(
  [(state) => state.transcripts.data, (_, videoId) => videoId],
  (data, videoId) => data[videoId]?.data
);

export const selectTranscriptStatus = createSelector(
  [(state) => state.transcripts.status],
  (status) => status
);

export default transcriptSlice.reducer;