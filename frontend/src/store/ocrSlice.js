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
    results: {},
    wordCloud: {},
    brandTable: {},
    processingStats: {},
    loading: false,
    error: null,
  },
  reducers: {
    setOcrResults: (state, action) => {
      const { id, results } = action.payload;
      state.results[id] = results;
    },
    setWordCloud: (state, action) => {
      const { id, wordCloud } = action.payload;
      state.wordCloud[id] = wordCloud;
    },
    setBrandTable: (state, action) => {
      const { id, brandTable } = action.payload;
      state.brandTable[id] = brandTable;
    },
    setProcessingStats: (state, action) => {
      const { id, stats } = action.payload;
      state.processingStats[id] = stats;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchOcrWordCloud.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchOcrWordCloud.fulfilled, (state, action) => {
        state.loading = false;
        state.wordCloud[action.payload.videoId] = { url: action.payload.wordCloudUrl };
        state.error = null;
      })
      .addCase(fetchOcrWordCloud.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload.error;
      })
      .addCase(fetchBrandsOcrTable.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchBrandsOcrTable.fulfilled, (state, action) => {
        state.loading = false;
        state.brandTable[action.payload.videoId] = { data: action.payload.brandTableData };
        state.error = null;
      })
      .addCase(fetchBrandsOcrTable.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload.error;
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