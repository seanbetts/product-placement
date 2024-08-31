import { configureStore } from '@reduxjs/toolkit';
import videoReducer from './videoSlice';
import transcriptReducer from './transcriptSlice';
import ocrReducer from './ocrSlice';

export const store = configureStore({
  reducer: {
    videos: videoReducer,
    transcripts: transcriptReducer,
    ocr: ocrReducer,
  },
});