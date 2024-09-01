import axios from 'axios';

export const handleApiError = (error) => {
  if (axios.isAxiosError(error)) {
    // Handle Axios errors
    const status = error.response?.status;
    const message = error.response?.data?.message || error.message;

    if (status === 404) {
      return new Error(`Resource not found: ${message}`);
    } else if (status === 401) {
      return new Error(`Unauthorized: ${message}`);
    } else if (status === 403) {
      return new Error(`Forbidden: ${message}`);
    } else if (status >= 500) {
      return new Error(`Server error: ${message}`);
    } else {
      return new Error(`API error: ${message}`);
    }
  } else {
    // Handle non-Axios errors
    return new Error(`Unexpected error: ${error.message}`);
  }
};