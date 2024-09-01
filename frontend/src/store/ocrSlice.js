import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../services/api';

export const fetchOcrWordCloud = createAsyncThunk(
  'ocr/fetchWordCloud',
  async (videoId, { rejectWithValue }) => {
    try {
      const wordCloudUrl = await api.getOcrWordCloud(videoId);
      return { videoId, wordCloudUrl };
    } catch (error) {
      return rejectWithValue({ videoId, error: error.message });
    }
  }
);

export const fetchBrandsOcrTable = createAsyncThunk(
  'ocr/fetchBrandsTable',
  async (videoId, { rejectWithValue }) => {
    try {
      const brandTableData = await api.getBrandsOcrTable(videoId);
      return { videoId, brandTableData };
    } catch (error) {
      return rejectWithValue({ videoId, error: error.message });
    }
  }
);

const ocrSlice = createSlice({
  name: 'ocr',
  initialState: {
    data: {
      results: {},
      wordCloud: {},
      brandTable: {},
      processingStats: {},
    },
    status: {
      loading: false,
      error: null,
    },
  },
  reducers: {
    setOcrResults: (state, action) => {
      const { id, results } = action.payload;
      state.data.results[id] = results;
    },
    setWordCloud: (state, action) => {
      const { id, wordCloud } = action.payload;
      state.data.wordCloud[id] = wordCloud;
    },
    setBrandTable: (state, action) => {
      const { id, brandTable } = action.payload;
      state.data.brandTable[id] = brandTable;
    },
    setProcessingStats: (state, action) => {
      const { id, stats } = action.payload;
      state.data.processingStats[id] = stats;
    },
    setLoading: (state, action) => {
      state.status.loading = action.payload;
    },
    setError: (state, action) => {
      state.status.error = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOcrWordCloud.pending, (state) => {
        state.status.loading = true;
      })
      .addCase(fetchOcrWordCloud.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data.wordCloud[action.payload.videoId] = { url: action.payload.wordCloudUrl };
        state.status.error = null;
      })
      .addCase(fetchOcrWordCloud.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload.error;
      })
      .addCase(fetchBrandsOcrTable.pending, (state) => {
        state.status.loading = true;
      })
      .addCase(fetchBrandsOcrTable.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data.brandTable[action.payload.videoId] = { data: action.payload.brandTableData };
        state.status.error = null;
      })
      .addCase(fetchBrandsOcrTable.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload.error;
      });
  },
});

export const { 
  setOcrResults, 
  setWordCloud, 
  setBrandTable, 
  setProcessingStats, 
  setLoading, 
  setError 
} = ocrSlice.actions;

export default ocrSlice.reducer;