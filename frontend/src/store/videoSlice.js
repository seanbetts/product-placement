import { createSlice } from '@reduxjs/toolkit';

const videoSlice = createSlice({
  name: 'videos',
  initialState: {
    list: [],
    details: {},
    loading: false,
    error: null,
    lastFetched: null,
  },
  reducers: {
    setVideos: (state, action) => {
      state.list = action.payload.videos;
      state.lastFetched = action.payload.lastFetched;
    },
    setVideoDetails: (state, action) => {
      state.details[action.payload.id] = action.payload.details;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
  },
});

export const { setVideos, setVideoDetails, setLoading, setError } = videoSlice.actions;
export default videoSlice.reducer;