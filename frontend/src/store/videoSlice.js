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
      return rejectWithValue(error.message);
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
    data: {
      list: [],
      details: {},
      frames: {},
      transcript: {},
      firstFrames: {},
    },
    status: {
      loading: false,
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
        state.data.list = action.payload;
      })
      .addCase(fetchProcessedVideos.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload;
      })
      .addCase(fetchFirstVideoFrame.fulfilled, (state, action) => {
        state.data.firstFrames[action.payload.videoId] = action.payload.imageData;
      })
      .addCase(fetchVideoDetails.pending, (state) => {
        state.status.loading = true;
        state.status.error = null;
      })
      .addCase(fetchVideoDetails.fulfilled, (state, action) => {
        state.status.loading = false;
        state.status.error = null;
        state.data.details[action.payload.video_id] = action.payload;
      })
      .addCase(fetchVideoDetails.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload;
      })
      .addCase(fetchVideoFrames.pending, (state) => {
        state.status.framesLoading = true;
      })
      .addCase(fetchVideoFrames.fulfilled, (state, action) => {
        state.status.framesLoading = false;
        state.data.frames[action.meta.arg] = {
          data: action.payload,
          lastFetched: Date.now()
        };
      })
      .addCase(fetchVideoFrames.rejected, (state, action) => {
        state.status.framesLoading = false;
        state.status.error = action.payload || 'Failed to fetch video frames';
      })
      .addCase(fetchTranscript.pending, (state, action) => {
        state.status.loading = true;
        state.status.error = null;
      })
      .addCase(fetchTranscript.fulfilled, (state, action) => {
        state.status.loading = false;
        state.data.transcript[action.meta.arg] = {
          data: action.payload,
          lastFetched: Date.now()
        };
      })
      .addCase(fetchTranscript.rejected, (state, action) => {
        state.status.loading = false;
        state.status.error = action.payload;
      })
      .addCase(updateVideoName.fulfilled, (state, action) => {
        const { videoId, newName } = action.meta.arg;
        if (state.data.details[videoId]) {
          state.data.details[videoId].name = newName;
        }
        state.ui.isEditingName = false;
      })
      .addCase(downloadFile.rejected, (state, action) => {
        state.ui.snackbar = { open: true, message: 'Failed to download file', severity: 'error' };
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