/**
 * PPO Adapter Service
 * 
 * Transforms PPO scheduler API responses to match the formats expected
 * by the existing front2 components designed for constraint programming.
 */

import { ScheduleResponse, ScheduledJob, Job } from './ppoTypes';

// Buffer status thresholds (in hours)
const BUFFER_THRESHOLDS = {
  LATE: 0,      // Already late
  WARNING: 24,  // Less than 24 hours
  CAUTION: 72,  // Less than 72 hours
  OK: Infinity  // More than 72 hours
};

// Calculate buffer status based on LCD date and current time
function calculateBufferStatus(lcdDate: string, currentTime: Date = new Date()): string {
  const lcd = new Date(lcdDate);
  const hoursUntilDue = (lcd.getTime() - currentTime.getTime()) / (1000 * 60 * 60);
  
  if (hoursUntilDue <= BUFFER_THRESHOLDS.LATE) return 'Late';
  if (hoursUntilDue <= BUFFER_THRESHOLDS.WARNING) return 'Warning';
  if (hoursUntilDue <= BUFFER_THRESHOLDS.CAUTION) return 'Caution';
  return 'OK';
}

// Calculate priority based on various factors
function calculatePriority(job: ScheduledJob, originalJob?: Job): number {
  // If we have the original job data, use its priority
  if (originalJob?.priority) {
    return originalJob.priority;
  }
  
  // Otherwise, infer priority from job ID patterns or other factors
  if (job.job_id.includes('URGENT') || job.job_id.includes('PRIORITY')) {
    return 1;
  }
  
  // Default priority
  return 3;
}

// Transform PPO scheduled jobs to Gantt Priority View format
export function transformToGanttPriorityView(
  scheduledJobs: ScheduledJob[], 
  jobs: Job[]
): any[] {
  const jobMap = new Map(jobs.map(j => [j.job_id, j]));
  
  // Track instance numbers for each job_id  
  const jobInstanceCount = new Map<string, number>();
  
  // Generate process codes for each unique job
  const jobProcessCodes = new Map<string, string>();
  const processCodePrefixes = ['CP', 'CD', 'CM', 'CV', 'CL', 'BL', 'GD', 'OV', 'AL'];
  let processCodeIndex = 0;
  let processNumberBase = 1;
  
  return scheduledJobs.map(scheduledJob => {
    const originalJob = jobMap.get(scheduledJob.job_id);
    const bufferStatus = originalJob ? calculateBufferStatus(originalJob.lcd_date) : 'OK';
    
    // Get or generate process code for this job
    if (!jobProcessCodes.has(scheduledJob.job_id)) {
      // Generate a process code like CP08-496
      const prefix = processCodePrefixes[processCodeIndex % processCodePrefixes.length];
      const number = String(processNumberBase).padStart(2, '0');
      const suffix = String(Math.floor(Math.random() * 900) + 100); // Random 3-digit number
      jobProcessCodes.set(scheduledJob.job_id, `${prefix}${number}-${suffix}`);
      
      processCodeIndex++;
      if (processCodeIndex % processCodePrefixes.length === 0) {
        processNumberBase++;
      }
    }
    
    const processCode = jobProcessCodes.get(scheduledJob.job_id)!;
    
    // Get instance number for this job
    const currentCount = jobInstanceCount.get(scheduledJob.job_id) || 0;
    const instanceNumber = currentCount + 1;
    jobInstanceCount.set(scheduledJob.job_id, instanceNumber);
    
    // Get total instances for this job
    const totalInstances = scheduledJobs.filter(sj => sj.job_id === scheduledJob.job_id).length;
    
    // Create a unique task identifier matching the required format
    // Format: JOBID_PROCESSCODE-INSTANCE/TOTAL (e.g., JOST25050298_CP08-496-1/3)
    const taskId = `${scheduledJob.job_id}_${processCode}-${instanceNumber}/${totalInstances}`;
    
    return {
      Task: taskId,
      Start: scheduledJob.start_datetime,
      Finish: scheduledJob.end_datetime,
      Resource: scheduledJob.machine_name,
      BufferStatusLabel: bufferStatus,
      PriorityLabel: `Priority ${calculatePriority(scheduledJob, originalJob)}`,
      JobFamily: originalJob?.family_id || scheduledJob.job_id.split('_')[0],
      ProcessNumber: originalJob?.sequence || 1,
      Color: null // Will use buffer status colors in the component
    };
  });
}

// Transform PPO scheduled jobs to Gantt Resource View format
export function transformToGanttResourceView(
  scheduledJobs: ScheduledJob[],
  jobs: Job[]
): any[] {
  const jobMap = new Map(jobs.map(j => [j.job_id, j]));
  
  // Generate process codes for all unique jobs (matching priority view)
  const jobProcessCodes = new Map<string, string>();
  const jobInstanceCount = new Map<string, number>();
  const processCodePrefixes = ['CP', 'CD', 'CM', 'CV', 'CL', 'BL', 'GD', 'OV', 'AL'];
  let processCodeIndex = 0;
  let processNumberBase = 1;
  
  // First pass: generate consistent process codes
  const uniqueJobIds = [...new Set(scheduledJobs.map(sj => sj.job_id))];
  uniqueJobIds.forEach(jobId => {
    const prefix = processCodePrefixes[processCodeIndex % processCodePrefixes.length];
    const number = String(processNumberBase).padStart(2, '0');
    const suffix = String(Math.floor(Math.random() * 900) + 100);
    jobProcessCodes.set(jobId, `${prefix}${number}-${suffix}`);
    
    processCodeIndex++;
    if (processCodeIndex % processCodePrefixes.length === 0) {
      processNumberBase++;
    }
  });
  
  // Group by machine for resource view
  const machineGroups = scheduledJobs.reduce((acc, job) => {
    if (!acc[job.machine_name]) {
      acc[job.machine_name] = [];
    }
    acc[job.machine_name].push(job);
    return acc;
  }, {} as Record<string, ScheduledJob[]>);
  
  const result: any[] = [];
  
  // Process all scheduled jobs to create unique task IDs
  scheduledJobs.forEach(scheduledJob => {
    const originalJob = jobMap.get(scheduledJob.job_id);
    const bufferStatus = originalJob ? calculateBufferStatus(originalJob.lcd_date) : 'OK';
    
    const processCode = jobProcessCodes.get(scheduledJob.job_id)!;
    const currentCount = jobInstanceCount.get(scheduledJob.job_id) || 0;
    const instanceNumber = currentCount + 1;
    jobInstanceCount.set(scheduledJob.job_id, instanceNumber);
    const totalInstances = scheduledJobs.filter(sj => sj.job_id === scheduledJob.job_id).length;
    
    const taskId = `${scheduledJob.job_id}_${processCode}-${instanceNumber}/${totalInstances}`;
    
    result.push({
      Task: taskId,
      Start: scheduledJob.start_datetime,
      Finish: scheduledJob.end_datetime,
      Resource: scheduledJob.machine_name,
      Machine: scheduledJob.machine_name,
      BufferStatusLabel: bufferStatus,
      PriorityLabel: `Priority ${calculatePriority(scheduledJob, originalJob)}`,
      JobFamily: originalJob?.family_id || scheduledJob.job_id.split('_')[0],
      ProcessNumber: originalJob?.sequence || 1,
      MachineUtilization: calculateMachineUtilization(Object.values(machineGroups).flat(), scheduledJob.machine_name)
    });
  });
  
  return result;
}

// Transform PPO scheduled jobs to Detailed Schedule Table format
export function transformToDetailedSchedule(
  scheduledJobs: ScheduledJob[],
  jobs: Job[]
): any[] {
  const jobMap = new Map(jobs.map(j => [j.job_id, j]));
  
  return scheduledJobs.map(scheduledJob => {
    const originalJob = jobMap.get(scheduledJob.job_id);
    const bufferStatus = originalJob ? calculateBufferStatus(originalJob.lcd_date) : 'OK';
    
    // Parse dates for display
    const startDate = new Date(scheduledJob.start_datetime);
    const endDate = new Date(scheduledJob.end_datetime);
    const duration = (endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60); // in hours
    
    return {
      op_id: scheduledJob.job_id,
      job_id: scheduledJob.job_id,
      job: originalJob?.family_id || scheduledJob.job_id.split('_')[0],
      process_code: `P${originalJob?.sequence || 1}`,
      MachineName_v: scheduledJob.machine_name,
      rsc_code: scheduledJob.machine_name,
      scheduled_start_time_str: scheduledJob.start_datetime,
      scheduled_end_time_str: scheduledJob.end_datetime,
      scheduled_start_date_str: formatDateString(startDate),
      scheduled_end_date_str: formatDateString(endDate),
      scheduled_start_time: formatTimeString(startDate),
      scheduled_end_time: formatTimeString(endDate),
      duration_hours: duration.toFixed(2),
      setup_hours: scheduledJob.setup_time_included || 0,
      processing_hours: duration - (scheduledJob.setup_time_included || 0),
      lcd_date_str: originalJob?.lcd_date || '',
      LCD_DATE: originalJob?.lcd_date || '',
      priority: calculatePriority(scheduledJob, originalJob),
      buffer_status: bufferStatus,
      job_quantity: originalJob?.processing_time ? Math.round(originalJob.processing_time * 100) : 100,
      expect_output_per_hour: 100, // Default value
      hours_need: originalJob?.processing_time || duration,
      number_operator: 1, // Default value
      is_scheduled: true,
      schedule_status: 'Scheduled'
    };
  });
}

// Create schedule overview from PPO metrics
export function createScheduleOverview(
  response: ScheduleResponse,
  jobs: Job[]
): any {
  const scheduledJobs = response.scheduled_jobs;
  const metrics = response.metrics;
  
  // Calculate buffer status distribution
  const bufferStatusCounts = { Late: 0, Warning: 0, Caution: 0, OK: 0 };
  const jobMap = new Map(jobs.map(j => [j.job_id, j]));
  
  scheduledJobs.forEach(scheduledJob => {
    const originalJob = jobMap.get(scheduledJob.job_id);
    if (originalJob) {
      const status = calculateBufferStatus(originalJob.lcd_date);
      bufferStatusCounts[status as keyof typeof bufferStatusCounts]++;
    }
  });
  
  // Find date range
  const dates = scheduledJobs.flatMap(job => [
    new Date(job.start_datetime).getTime(),
    new Date(job.end_datetime).getTime()
  ]);
  const minDate = Math.min(...dates);
  const maxDate = Math.max(...dates);
  
  return {
    total_jobs: metrics.total_jobs,
    scheduled_jobs: metrics.scheduled_jobs,
    completion_rate: metrics.completion_rate,
    makespan_hours: metrics.makespan,
    average_utilization: metrics.average_utilization,
    buffer_status_distribution: bufferStatusCounts,
    date_range: {
      start: new Date(minDate).toISOString(),
      end: new Date(maxDate).toISOString()
    },
    important_jobs_on_time: metrics.important_jobs_on_time,
    total_setup_time: metrics.total_setup_time,
    schedule_id: response.schedule_id,
    generated_at: response.timestamp
  };
}

// Helper function to calculate machine utilization
function calculateMachineUtilization(machineJobs: ScheduledJob[], machineName: string): number {
  if (machineJobs.length === 0) return 0;
  
  // Sort jobs by start time
  const sortedJobs = [...machineJobs].sort((a, b) => 
    new Date(a.start_datetime).getTime() - new Date(b.start_datetime).getTime()
  );
  
  // Calculate total working time
  let totalWorkTime = 0;
  sortedJobs.forEach(job => {
    const duration = (new Date(job.end_datetime).getTime() - 
                     new Date(job.start_datetime).getTime()) / (1000 * 60 * 60);
    totalWorkTime += duration;
  });
  
  // Calculate total span time (from first job start to last job end)
  const firstStart = new Date(sortedJobs[0].start_datetime).getTime();
  const lastEnd = new Date(sortedJobs[sortedJobs.length - 1].end_datetime).getTime();
  const totalSpanTime = (lastEnd - firstStart) / (1000 * 60 * 60);
  
  return totalSpanTime > 0 ? (totalWorkTime / totalSpanTime) * 100 : 0;
}

// Format date string for display
function formatDateString(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

// Format time string for display
function formatTimeString(date: Date): string {
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  return `${hours}:${minutes}`;
}

// Transform system logs (if needed)
export function transformSystemLogs(logs: any[]): any[] {
  // For now, just pass through the logs
  // Add transformation logic if needed
  return logs || [];
}