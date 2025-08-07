import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const schedulingApi = {
  // Schedule jobs with selected dataset and model
  scheduleJobs: async (dataset = '40_jobs', model = 'sb3_1million') => {
    try {
      const response = await api.post('/schedule', {
        dataset,
        model,
        deterministic: true,
        max_steps: 10000
      });
      return response.data;
    } catch (error) {
      console.error('Scheduling error:', error);
      throw error;
    }
  },

  // Get available datasets
  getDatasets: async () => {
    try {
      const response = await api.get('/datasets');
      return response.data;
    } catch (error) {
      console.error('Error fetching datasets:', error);
      throw error;
    }
  },

  // Get available models
  getModels: async () => {
    try {
      const response = await api.get('/models');
      return response.data;
    } catch (error) {
      console.error('Error fetching models:', error);
      throw error;
    }
  }
};

export default schedulingApi;