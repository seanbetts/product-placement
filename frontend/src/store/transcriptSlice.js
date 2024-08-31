import { createSlice } from '@reduxjs/toolkit';

const transcriptSlice = createSlice({
  name: 'transcripts',
  initialState: {
    data: {},
    loading: false,
    error: null,
  },
  reducers: {
    setTranscript: (state, action) => {
      state.data[action.payload.id] = action.payload.transcript;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
  },
});

export const { setTranscript, setLoading, setError } = transcriptSlice.actions;
export default transcriptSlice.reducer;