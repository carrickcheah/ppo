import axios from 'axios';
import { 
  HealthResponse, 
  ScheduleRequest, 
  ScheduleResponse,
  Job 
} from '../types';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY || 'dev-api-key-change-in-production';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// API service methods
export const api = {
  // Check API health
  async getHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },

  // Create a new schedule
  async createSchedule(request: ScheduleRequest): Promise<ScheduleResponse> {
    const response = await apiClient.post<ScheduleResponse>('/schedule', request);
    return response.data;
  },

  // Get sample jobs for testing
  generateSampleJobs(count: number = 20): Job[] {
    const jobs: Job[] = [];
    const prefixes = ['JOAW', 'JOST', 'JOEX'];
    const machineTypeSets = [[1, 2, 3], [2, 3, 4], [1, 3], [2, 4], [1, 2, 3, 4]];
    
    for (let i = 0; i < count; i++) {
      const now = new Date();
      const lcdDate = new Date(now.getTime() + (1 + i % 7) * 24 * 60 * 60 * 1000);
      
      jobs.push({
        job_id: `${prefixes[i % prefixes.length]}${String(i).padStart(4, '0')}`,
        family_id: `FAM${String(Math.floor(i / 5)).padStart(3, '0')}`,
        sequence: (i % 5) + 1,
        processing_time: 1.5 + (i % 10) * 0.3,
        machine_types: machineTypeSets[i % machineTypeSets.length],
        priority: (i % 3) + 1,
        is_important: i % 4 === 0,
        lcd_date: lcdDate.toISOString(),
        setup_time: i % 2 === 0 ? 0.3 : 0.5,
      });
    }
    
    return jobs;
  },

  // Load jobs from JSON file
  async loadJobsFromFile(file: File): Promise<Job[]> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          const data = JSON.parse(content);
          const jobs = Array.isArray(data) ? data : data.jobs || [];
          resolve(jobs);
        } catch (error) {
          reject(new Error('Invalid JSON file'));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  },
};

// Error handler
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.status, error.response.data);
      
      if (error.response.status === 401 || error.response.status === 403) {
        throw new Error('Authentication failed. Please check API key.');
      } else if (error.response.status === 500) {
        throw new Error('Server error. Please check database connection.');
      }
    } else if (error.request) {
      // No response received
      console.error('No response from server:', error.request);
      throw new Error('Cannot connect to API server. Please ensure it is running.');
    }
    
    throw error;
  }
);

export default api;