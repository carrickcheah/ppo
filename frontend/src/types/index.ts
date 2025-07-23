// Type definitions for PPO Scheduler

export interface Job {
  job_id: string;
  family_id: string;
  sequence: number;
  processing_time: number;
  machine_types: number[];
  priority: number;
  is_important: boolean;
  lcd_date: string;
  setup_time: number;
}

export interface ScheduledJob {
  job_id: string;
  machine_id: number;
  machine_name: string;
  start_time: number;
  end_time: number;
  start_datetime: string;
  end_datetime: string;
  setup_time_included: number;
}

export interface ScheduleMetrics {
  makespan: number;
  total_jobs: number;
  scheduled_jobs: number;
  completion_rate: number;
  average_utilization: number;
  total_setup_time: number;
  important_jobs_on_time: number;
}

export interface ScheduleRequest {
  request_id?: string;
  jobs: Job[];
  schedule_start?: string;
  machines?: Machine[];
  save_to_database?: boolean;
}

export interface ScheduleResponse {
  schedule_id: string;
  scheduled_jobs: ScheduledJob[];
  metrics: ScheduleMetrics;
  timestamp: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  model_loaded: boolean;
  database_connected: boolean;
  uptime: number;
  last_schedule_time: string | null;
  environment: string;
}

export interface Machine {
  machine_id: number;
  machine_name: string;
  machine_type: number;
  current_load?: number;
}

export interface GanttData {
  jobId: string;
  machineName: string;
  startTime: number;
  endTime: number;
  duration: number;
  color: string;
}