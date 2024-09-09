import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../services/api';

export const fetchProcessedVideos = createAsyncThunk(
  'videos/fetchProcessedVideos',
  async (_, { getState, rejectWithValue }) => {
    const state = getState().videos;
    const now = Date.now();
    if (state.status.lastFetched && now - state.status.lastFetched < 60000) { // 1 minute cache
      return { videos: state.data.list, lastFetched: state.status.lastFetched };
    }
    try {
      const videos = await api.getProcessedVideos();
      return { videos, lastFetched: now };
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchVideoStatus = createAsyncThunk(
  'videos/fetchVideoStatus',
  async (_, { rejectWithValue }) => {
    try {
      return await api.getVideoStatus();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchVideoDetails = createAsyncThunk(
  'videos/fetchVideoDetails',
  async (videoId, { rejectWithValue }) => {
    try {
      return await api.getProcessingStats(videoId);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const fetchFirstVideoFrame = createAsyncThunk(
  'videos/fetchFirstVideoFrame',
  async (videoId, { rejectWithValue }) => {
    try {
      const imageData = await api.getFirstVideoFrame(videoId);
      return { videoId, imageData };
    } catch (error) {
      console.error(`Error fetching first frame for video ${videoId}:`, error);
      return rejectWithValue({ videoId, error: error.message });
    }
  }
);

export const fetchVideoFrames = createAsyncThunk(
  'videos/fetchVideoFrames',
  async (videoId, { rejectWithValue }) => {
    try {
      const frames = await api.getAllVideoFrames(videoId);
      if (!frames || frames.length === 0) {
        return rejectWithValue('No frames available for this video');
      }
      return frames;
    } catch (error) {
      return rejectWithValue(error.message || 'Failed to fetch video frames');
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

const videoSlice = createSlice({
  name: 'videos',
  initialState: {
    data: {
      list: [],
      details: {},
      frames: {},
      firstFrames: {},
    },
    status: {
      loading: false,
      loadingDetails: {},
      loadingFrames: {},
      error: null,
      lastFetched: null,
      framesLoading: false,
    },
    ui: {
      searchTerm: '',
      isEditingName: false,
      editingName: '',
      snackbar: { open: false, message: '', severity: 'success' },
    },
  },
  reducers: {
    setVideos: (state, action) => {
      state.data.list = action.payload.videos;
      state.status.lastFetched = action.payload.lastFetched;
    },
    setVideoFrames: (state, action) => {
      state.data.frames[action.payload.id] = action.payload.frames;
    },
    setLoading: (state, action) => {
      state.status.loading = action.payload;
    },
    setError: (state, action) => {
      state.status.error = action.payload;
    },
    setSearchTerm: (state, action) => {
      state.ui.searchTerm = action.payload;
    },
    setIsEditingName: (state, action) => {
      state.ui.isEditingName = action.payload;
    },
    setEditingName: (state, action) => {
      state.ui.editingName = action.payload;
    },
    setSnackbar: (state, action) => {
      state.ui.snackbar = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchProcessedVideos.pending, (state) => {
        state.status.loading = true;
      })
      .addCase(fetchProcessedVideos.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data.list = action.payload.videos;
        state.status.lastFetched = action.payload.lastFetched;
      })
      .addCase(fetchProcessedVideos.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload;
      })
      .addCase(fetchFirstVideoFrame.fulfilled, (state, action) => {
        state.data.firstFrames[action.payload.videoId] = action.payload.imageData;
      })
      .addCase(fetchFirstVideoFrame.rejected, (state, action) => {
        state.data.firstFrames[action.payload.videoId] = null;
      })
      .addCase(fetchVideoDetails.pending, (state, action) => {
        state.status.loadingDetails[action.meta.arg] = true;
        state.status.error = null;
      })
      .addCase(fetchVideoDetails.fulfilled, (state, action) => {
        state.status.loadingDetails[action.meta.arg] = false;
        state.status.error = null;
        state.data.details[action.payload.video_id] = action.payload;
      })
      .addCase(fetchVideoDetails.rejected, (state, action) => {
        state.status.loadingDetails[action.meta.arg] = false;
        state.status.error = action.payload;
      })
      .addCase(fetchVideoFrames.pending, (state, action) => {
        state.status.loadingFrames[action.meta.arg] = true;
      })
      .addCase(fetchVideoFrames.fulfilled, (state, action) => {
        state.status.loadingFrames[action.meta.arg] = false;
        state.data.frames[action.meta.arg] = {
          data: action.payload,
          lastFetched: Date.now()
        };
      })
      .addCase(fetchVideoFrames.rejected, (state, action) => {
        state.status.loadingFrames[action.meta.arg] = false;
        state.status.error = action.payload || 'Failed to fetch video frames';
      })
      .addCase(updateVideoName.fulfilled, (state, action) => {
        if (state.data.details[action.meta.arg.videoId]) {
          state.data.details[action.meta.arg.videoId].name = action.meta.arg.newName;
        }
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

export const selectVideoDetails = (state, videoId) => state.videos.data.details[videoId];
export const selectVideoFrames = (state, videoId) => state.videos.data.frames[videoId]?.data;
export const selectVideoLoadingStates = (state, videoId) => ({
  loadingDetails: state.videos.status.loadingDetails[videoId],
  loadingFrames: state.videos.status.loadingFrames[videoId],
});

export default videoSlice.reducer;