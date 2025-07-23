/**
 * PPO API Service
 * 
 * Handles communication with the PPO scheduler backend API.
 * This service replaces the constraint programming API calls.
 */

import { 
  HealthResponse, 
  ScheduleRequest, 
  ScheduleResponse,
  Job,
  Machine
} from './ppoTypes';
import { PPO_API_URL } from '../config';

// Use PPO API URL from config
const API_BASE_URL = PPO_API_URL;

// API service class
class PPOApiService {
  private baseUrl: string;
  private defaultHeaders: HeadersInit;
  
  constructor() {
    this.baseUrl = API_BASE_URL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }
  
  /**
   * Check API health status
   */
  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: this.defaultHeaders,
      });
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }
  
  /**
   * Create a new schedule using PPO model
   * 
   * @param jobs - Optional array of jobs. If empty, loads from database
   * @param machines - Optional array of machines. If not provided, loads from database
   * @param scheduleStart - Optional start time for schedule. Defaults to current time
   * @param saveToDatabase - Whether to save the generated schedule to database
   */
  async createSchedule(
    jobs?: Job[],
    machines?: Machine[],
    scheduleStart?: Date,
    saveToDatabase: boolean = false
  ): Promise<ScheduleResponse> {
    try {
      const request: ScheduleRequest = {
        request_id: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        jobs: jobs || [],
        machines: machines,
        schedule_start: scheduleStart?.toISOString() || new Date().toISOString(),
        respect_break_times: true,
        respect_holidays: true,
        optimization_objective: 'makespan',
        save_to_database: saveToDatabase,
      };
      
      console.log('Creating schedule with request:', {
        jobCount: request.jobs.length,
        machineCount: request.machines?.length,
        scheduleStart: request.schedule_start,
      });
      
      const response = await fetch(`${this.baseUrl}/schedule`, {
        method: 'POST',
        headers: this.defaultHeaders,
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const errorMessage = errorData?.detail || `Schedule creation failed: ${response.status}`;
        throw new Error(errorMessage);
      }
      
      const scheduleData = await response.json();
      console.log('Schedule created successfully:', {
        scheduleId: scheduleData.schedule_id,
        scheduledJobs: scheduleData.scheduled_jobs.length,
        metrics: scheduleData.metrics,
      });
      
      return scheduleData;
    } catch (error) {
      console.error('Schedule creation error:', error);
      throw error;
    }
  }
  
  /**
   * Load pending jobs from database and create schedule
   * This is a convenience method that calls createSchedule with no jobs
   */
  async scheduleFromDatabase(
    machines?: Machine[],
    scheduleStart?: Date,
    saveToDatabase: boolean = true
  ): Promise<ScheduleResponse> {
    return this.createSchedule([], machines, scheduleStart, saveToDatabase);
  }
  
  /**
   * Get recent system logs - PPO backend doesn't support this
   * Always returns empty logs array
   */
  async getSystemLogs(lines: number = 500): Promise<{ logs: any[] }> {
    // PPO backend doesn't have logs endpoint
    return { logs: [] };
  }
  
  /**
   * Generate sample jobs for testing
   * This creates production-like jobs that are compatible with the PPO model
   */
  generateSampleJobs(count: number = 200): Job[] {
    const jobs: Job[] = [];
    const prefixes = ['JOAW', 'JOST', 'JOEX', 'JOTP', 'JOCF', 'JOCH', 'JOCM', 'JOCP'];
    const currentDate = new Date();
    
    // Generate production-like jobs
    const numFamilies = Math.floor(count / 4); // Average 4 jobs per family
    
    for (let f = 0; f < numFamilies; f++) {
      const prefix = prefixes[f % prefixes.length];
      const familyId = `${prefix}${String(25060000 + f * 100).slice(-7)}`;
      const numJobsInFamily = 3 + Math.floor(Math.random() * 3); // 3-5 jobs per family
      const baseLcdDays = Math.floor(Math.random() * 10) + 3; // 3-12 days out
      const isImportantFamily = Math.random() > 0.8; // 20% are important
      
      for (let j = 0; j < numJobsInFamily && jobs.length < count; j++) {
        const lcdDate = new Date(currentDate.getTime() + (baseLcdDays + j * 0.5) * 24 * 60 * 60 * 1000);
        
        jobs.push({
          job_id: `${familyId}_${prefix.slice(2)}-${String(j + 1).padStart(3, '0')}-${j + 1}`,
          family_id: familyId,
          sequence: j + 1,
          processing_time: 1.5 + Math.random() * 8.5, // 1.5-10 hours
          machine_types: this.getRandomMachineTypes(),
          priority: isImportantFamily ? 1 : (Math.random() > 0.5 ? 2 : 3),
          is_important: isImportantFamily && j === 0, // First job in important family
          lcd_date: lcdDate.toISOString(),
          setup_time: 0.3 + Math.random() * 0.7, // 0.3-1.0 hours
        });
      }
    }
    
    return jobs;
  }
  
  /**
   * Get random machine types compatible with PPO model
   */
  private getRandomMachineTypes(): number[] {
    const machineTypeSets = [
      [1, 2, 3],
      [2, 3, 4],
      [1, 3],
      [2, 4],
      [1, 2, 3, 4],
      [1],
      [2],
      [3],
      [4]
    ];
    
    return machineTypeSets[Math.floor(Math.random() * machineTypeSets.length)];
  }
}

// Export singleton instance
export const ppoApi = new PPOApiService();

// Also export the class for testing purposes
export default PPOApiService;