import { useState, useEffect } from 'react';

export interface WorkingHour {
  start_time: string;
  end_time: string;
  is_working: boolean;
}

export interface BreakTime {
  name: string;
  description: string;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  break_type: string;
  is_mandatory: boolean;
}

export interface Holiday {
  date: string;
  name: string;
  description: string;
  scope: string;
  is_recurring: boolean;
}

export interface WorkingHoursConfig {
  working_hours_by_day: Record<string, WorkingHour[]>;
  break_times: BreakTime[];
  holidays: Holiday[];
  environment_config: {
    normal_working_hours: number;
    ot_working_hours: number;
    emergency_ot_hours: number;
    timezone: string;
  };
  cache_info: {
    last_refreshed: string;
    working_days_count: number;
    break_times_count: number;
    holidays_count: number;
  };
}

export interface UseWorkingHoursResult {
  config: WorkingHoursConfig | null;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export const useWorkingHours = (): UseWorkingHoursResult => {
  const [config, setConfig] = useState<WorkingHoursConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkingHours = async () => {
    setIsLoading(true);
    setError(null);
    
    // PPO backend doesn't have working hours endpoint
    // Use default working hours configuration
    const defaultWorkingHours: WorkingHoursConfig = {
      working_hours_by_day: {
        Monday: [{ start_time: "06:30:00", end_time: "23:59:59", is_working: true }],
        Tuesday: [{ start_time: "06:30:00", end_time: "23:59:59", is_working: true }],
        Wednesday: [{ start_time: "06:30:00", end_time: "23:59:59", is_working: true }],
        Thursday: [{ start_time: "06:30:00", end_time: "23:59:59", is_working: true }],
        Friday: [{ start_time: "06:30:00", end_time: "23:59:59", is_working: true }],
        Saturday: [{ start_time: "06:30:00", end_time: "13:00:00", is_working: true }],
        Sunday: [{ start_time: "00:00:00", end_time: "00:00:00", is_working: false }]
      },
      break_times: [
        { name: "Morning Tea", description: "Morning tea break", start_time: "09:45:00", end_time: "10:00:00", duration_minutes: 15, break_type: "tea", is_mandatory: true },
        { name: "Lunch", description: "Lunch break", start_time: "12:45:00", end_time: "13:30:00", duration_minutes: 45, break_type: "lunch", is_mandatory: true },
        { name: "Afternoon Tea", description: "Afternoon tea break", start_time: "15:15:00", end_time: "15:30:00", duration_minutes: 15, break_type: "tea", is_mandatory: true }
      ],
      holidays: [],
      environment_config: {
        normal_working_hours: 8,
        ot_working_hours: 4,
        emergency_ot_hours: 2,
        timezone: "Asia/Singapore"
      },
      cache_info: {
        last_refreshed: new Date().toISOString(),
        working_days_count: 6,
        break_times_count: 3,
        holidays_count: 0
      }
    };
    
    setConfig(defaultWorkingHours);
    
    // Cache the data in localStorage
    try {
      localStorage.setItem('workingHoursCache', JSON.stringify({
        data: defaultWorkingHours,
        timestamp: Date.now()
      }));
    } catch (cacheError) {
      console.warn('Failed to cache working hours data:', cacheError);
    }
    
    setIsLoading(false);
  };

  const refresh = async () => {
    await fetchWorkingHours();
  };

  useEffect(() => {
    // Load from cache first
    try {
      const cached = localStorage.getItem('workingHoursCache');
      if (cached) {
        const { data, timestamp } = JSON.parse(cached);
        // Use cache if less than 1 hour old
        if (Date.now() - timestamp < 3600000) {
          setConfig(data);
          console.log('📦 Loaded working hours from cache');
          return;
        }
      }
    } catch (cacheError) {
      console.warn('Failed to load cached working hours:', cacheError);
    }
    
    // Fetch fresh data if no valid cache
    fetchWorkingHours();
  }, []);

  return {
    config,
    isLoading,
    error,
    refresh
  };
};

// Helper function to convert time string to minutes since midnight
export const timeToMinutes = (timeString: string): number => {
  const [hours, minutes, seconds] = timeString.split(':').map(Number);
  return hours * 60 + minutes + (seconds || 0) / 60;
};

// Helper function to convert minutes since midnight to time string
export const minutesToTime = (minutes: number): string => {
  const hours = Math.floor(minutes / 60);
  const mins = Math.floor(minutes % 60);
  const secs = Math.floor((minutes % 1) * 60);
  return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

// Helper function to check if a time is within a working period
export const isTimeInWorkingPeriod = (
  timeMinutes: number,
  workingHours: WorkingHour[]
): boolean => {
  return workingHours.some(period => {
    if (!period.is_working) return false;
    const startMinutes = timeToMinutes(period.start_time);
    const endMinutes = timeToMinutes(period.end_time);
    return timeMinutes >= startMinutes && timeMinutes < endMinutes;
  });
};

// Helper function to check if a time is within a break period
export const isTimeInBreak = (
  timeMinutes: number,
  breakTimes: BreakTime[]
): BreakTime | null => {
  for (const breakTime of breakTimes) {
    const startMinutes = timeToMinutes(breakTime.start_time);
    const endMinutes = timeToMinutes(breakTime.end_time);
    if (timeMinutes >= startMinutes && timeMinutes < endMinutes) {
      return breakTime;
    }
  }
  return null;
};