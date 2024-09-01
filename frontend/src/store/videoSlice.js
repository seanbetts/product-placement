import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../services/api';

export const fetchProcessedVideos = createAsyncThunk(
  'videos/fetchProcessedVideos',
  async (_, { rejectWithValue }) => {
    try {
      return await api.getProcessedVideos();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchVideoDetails = createAsyncThunk(
  'videos/fetchVideoDetails',
  async (videoId, { rejectWithValue }) => {
    try {
      return await api.getVideoDetails(videoId);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchVideoFrames = createAsyncThunk(
  'videos/fetchVideoFrames',
  async (videoId, { rejectWithValue }) => {
    try {
      return await api.getVideoFrames(videoId);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchTranscript = createAsyncThunk(
  'videos/fetchTranscript',
  async (videoId, { rejectWithValue }) => {
    try {
      return await api.getTranscript(videoId);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const updateVideoName = createAsyncThunk(
  'videos/updateVideoName',
  async ({ videoId, newName }, { rejectWithValue }) => {
    try {
      return await api.updateVideoName(videoId, newName);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const downloadFile = createAsyncThunk(
  'videos/downloadFile',
  async ({ videoId, fileType }, { rejectWithValue }) => {
    try {
      return await api.downloadFile(videoId, fileType);
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
      .addCase(fetchProcessedVideos.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchProcessedVideos.fulfilled, (state, action) => {
        state.loading = false;
        state.list = action.payload;
      })
      .addCase(fetchProcessedVideos.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
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