import axios from 'axios';

export const handleApiError = (error) => {
  if (axios.isAxiosError(error)) {
    // Handle Axios errors
    const status = error.response?.status;
    const message = error.response?.data?.message || error.message;
    
    switch (status) {
      case 400:
        return new Error(`Bad request: ${message}. Please check your input.`);
      case 401:
        return new Error(`Unauthorized: ${message}. Please log in again.`);
      case 403:
        if (message.toLowerCase().includes('api key')) {
          return new Error(`Authentication failed: Invalid or missing API key. Please check your API key.`);
        }
        return new Error(`Forbidden: ${message}. You don't have permission to access this resource.`);
      case 404:
        return new Error(`Resource not found: ${message}. Please check the URL.`);
      case 429:
        return new Error(`Too many requests: ${message}. Please try again later.`);
      case 500:
        return new Error(`Internal Server Error (500): ${message}. The server encountered an unexpected condition. Please try again later or contact support.`);
      case 501:
        return new Error(`Not Implemented (501): ${message}. The server does not support the functionality required to fulfill the request. Please check if you're using the correct API version.`);
      case 502:
        return new Error(`Bad Gateway (502): ${message}. The server received an invalid response from an upstream server. This is often a temporary error, please try again in a few minutes.`);
      case 503:
        return new Error(`Service Unavailable (503): ${message}. The server is temporarily unable to handle the request. This is often due to maintenance or overloading. Please try again later.`);
      case 504:
        return new Error(`Gateway Timeout (504): ${message}. The server did not receive a timely response from an upstream server. Please check your network connection and try again.`);
      default:
        return new Error(`API error (${status}): ${message}. Please try again or contact support.`);
    }
  } else {
    // Handle non-Axios errors
    return new Error(`Unexpected error: ${error.message}. Please try again or contact support.`);
  }
};