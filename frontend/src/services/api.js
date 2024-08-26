import axios from 'axios';

const API_BASE_URL = 'https://product-placement-pidup46kxa-nw.a.run.app';

export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('video', file);

  const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const getVideoStatus = async (videoId) => {
  const response = await axios.get(`${API_BASE_URL}/status/${videoId}`);
  return response.data;
};