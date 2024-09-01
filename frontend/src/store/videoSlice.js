import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../services/api';

export const fetchVideoDetails = createAsyncThunk(
  'videos/fetchVideoDetails',
  async (videoId, { dispatch, getState, rejectWithValue }) => {
    try {
      return await api.getVideoDetails(videoId, dispatch, getState);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchVideoFrames = createAsyncThunk(
  'videos/fetchVideoFrames',
  async (videoId, { dispatch, getState, rejectWithValue }) => {
    try {
      return await api.getVideoFrames(videoId, dispatch, getState);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchTranscript = createAsyncThunk(
  'videos/fetchTranscript',
  async (videoId, { dispatch, getState, rejectWithValue }) => {
    try {
      return await api.getTranscript(videoId, dispatch, getState);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const updateVideoName = createAsyncThunk(
  'videos/updateVideoName',
  async ({ videoId, newName }, { dispatch, rejectWithValue }) => {
    try {
      return await api.updateVideoName(videoId, newName, dispatch);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const downloadFile = createAsyncThunk(
  'videos/downloadFile',
  async ({ videoId, fileType }, { dispatch, rejectWithValue }) => {
    try {
      return await api.downloadFile(videoId, fileType, dispatch);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const videoSlice = createSlice({
  name: 'videos',
  initialState: {
    list: [],
    details: {},
    frames: {},
    transcript: {},
    loading: false,
    error: null,
    lastFetched: null,
    searchTerm: '',
    isEditingName: false,
    editingName: '',
    snackbar: { open: false, message: '', severity: 'success' },
  },
  reducers: {
    setVideos: (state, action) => {
      state.list = action.payload.videos;
      state.lastFetched = action.payload.lastFetched;
    },
    setVideoFrames: (state, action) => {
      state.frames[action.payload.id] = action.payload.frames;
    },
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
    setSearchTerm: (state, action) => {
      state.searchTerm = action.payload;
    },
    setIsEditingName: (state, action) => {
      state.isEditingName = action.payload;
    },
    setEditingName: (state, action) => {
      state.editingName = action.payload;
    },
    setSnackbar: (state, action) => {
      state.snackbar = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchVideoDetails.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchVideoDetails.fulfilled, (state, action) => {
        state.loading = false;
        state.error = null;
        state.details[action.payload.video_id] = action.payload;
      })
      .addCase(fetchVideoDetails.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      .addCase(fetchVideoFrames.fulfilled, (state, action) => {
        state.frames[action.meta.arg] = {
          data: action.payload,
          lastFetched: Date.now()
        };
      })
      .addCase(fetchTranscript.fulfilled, (state, action) => {
        state.transcript[action.meta.arg] = {
          data: action.payload,
          lastFetched: Date.now()
        };
      })
      .addCase(updateVideoName.fulfilled, (state, action) => {
        const { videoId, newName } = action.meta.arg;
        if (state.details[videoId]) {
          state.details[videoId].name = newName;
        }
        state.isEditingName = false;
      })
      .addCase(downloadFile.rejected, (state, action) => {
        state.snackbar = { open: true, message: 'Failed to download file', severity: 'error' };
      });
  },
});

export const { 
  setVideos, 
  setVideoFrames,
  setLoading,
  setError,
  setSearchTerm, 
  setIsEditingName, 
  setEditingName, 
  setSnackbar 
} = videoSlice.actions;

export default videoSlice.reducer;