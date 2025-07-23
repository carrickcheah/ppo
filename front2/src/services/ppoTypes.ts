/**
 * Type definitions for PPO Scheduler API
 * 
 * These types match the PPO backend models defined in /app/src/deployment/models.py
 */

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

export interface Machine {
  machine_id: number;
  machine_name: string;
  machine_type: number;
  status?: 'available' | 'busy' | 'maintenance' | 'offline';
  current_load?: number;
}

export interface ScheduledJob {
  job_id: string;
  machine_id: number;
  machine_name: string;
  start_time: number;  // Hours from schedule start
  end_time: number;    // Hours from schedule start
  start_datetime: string;  // ISO datetime string
  end_datetime: string;    // ISO datetime string
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
  machines?: Machine[];
  schedule_start: string;
  respect_break_times?: boolean;
  respect_holidays?: boolean;
  optimization_objective?: string;
  save_to_database?: boolean;
}

export interface ScheduleResponse {
  schedule_id: string;
  scheduled_jobs: ScheduledJob[];
  metrics: ScheduleMetrics;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_type: string;
  scheduler_type: string;
  database_connected: boolean;
  uptime_seconds: number;
  last_schedule_time: string | null;
  version: string;
}

export interface ErrorResponse {
  detail: string;
  error_code?: string;
  timestamp: string;
}