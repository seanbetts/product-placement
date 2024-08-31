import { createSlice } from '@reduxjs/toolkit';

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
      state.results[action.payload.id] = action.payload.results;
    },
    setWordCloud: (state, action) => {
      state.wordCloud[action.payload.id] = action.payload.wordCloud;
    },
    setBrandTable: (state, action) => {
      state.brandTable[action.payload.id] = action.payload.brandTable;
    },
    setProcessingStats: (state, action) => {
      state.processingStats[action.payload.id] = action.payload.stats;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
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