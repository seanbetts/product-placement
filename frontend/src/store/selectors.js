import { createSelector } from '@reduxjs/toolkit';

export const selectFilteredVideos = createSelector(
  state => state.videos.list,
  state => state.videos.searchTerm,
  state => state.videos.startDate,
  state => state.videos.endDate,
  state => state.videos.sortCriteria,
  state => state.videos.sortOrder,
  (videos, searchTerm, startDate, endDate, sortCriteria, sortOrder) => {
    if (!Array.isArray(videos)) {
      console.error('videos is not an array:', videos);
      return [];
    }

    let result = videos.filter(video => 
      (video.details?.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
       video.video_id.toLowerCase().includes(searchTerm.toLowerCase())) &&
      (!startDate || new Date(video.details?.total_processing_end_time) >= startDate) &&
      (!endDate || new Date(video.details?.total_processing_end_time) <= endDate)
    );

    result.sort((a, b) => {
      let comparison = 0;
      switch (sortCriteria) {
        case 'date':
          comparison = new Date(b.details?.total_processing_end_time).getTime() - new Date(a.details?.total_processing_end_time).getTime();
          break;
        case 'length':
          comparison = parseFloat(b.details?.video_length) - parseFloat(a.details?.video_length);
          break;
        default:
          comparison = 0;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return result;
  }
);