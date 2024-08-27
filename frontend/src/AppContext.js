import React, { createContext, useState, useContext } from 'react';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [videoId, setVideoId] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);

  return (
    <AppContext.Provider value={{ videoId, setVideoId, uploadStatus, setUploadStatus }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  return useContext(AppContext);
}