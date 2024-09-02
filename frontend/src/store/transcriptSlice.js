import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../services/api';

export const fetchTranscript = createAsyncThunk(
  'transcripts/fetchTranscript',
  async (videoId, { rejectWithValue }) => {
    try {
      return await api.getTranscript(videoId);
    } catch (error) {
      return rejectWithValue(error.message);
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
    },
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchTranscript.pending, (state) => {
        state.status.loading = true;
        state.status.error = null;
      })
      .addCase(fetchTranscript.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data[action.meta.arg] = {
          data: action.payload,
          lastFetched: Date.now()
        };
      })
      .addCase(fetchTranscript.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload;
      });
  },
});

export default transcriptSlice.reducer;