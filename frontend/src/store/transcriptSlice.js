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
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchTranscript.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTranscript.fulfilled, (state, action) => {
        state.loading = false;
        state.data[action.meta.arg] = action.payload;
      })
      .addCase(fetchTranscript.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export default transcriptSlice.reducer;