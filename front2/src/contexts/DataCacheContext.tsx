import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ppoApi } from '../services/ppoApi';
import { 
  transformToGanttPriorityView,
  transformToGanttResourceView,
  transformToDetailedSchedule,
  createScheduleOverview
} from '../services/ppoAdapter';
import { Job } from '../services/ppoTypes';

interface CachedData {
  ganttPriorityView: any[];
  ganttResourceView: any[];
  detailedSchedule: any[];
  scheduleOverview: any;
  isLoading: boolean;
  error: string | null;
  lastRefresh: Date;
}

interface DataCacheContextType {
  data: CachedData;
  refreshData: () => Promise<void>;
  clearError: () => void;
  clearCache: () => void;
}

const DataCacheContext = createContext<DataCacheContextType | undefined>(undefined);

export const useDataCache = () => {
  const context = useContext(DataCacheContext);
  if (context === undefined) {
    throw new Error('useDataCache must be used within a DataCacheProvider');
  }
  return context;
};

interface DataCacheProviderProps {
  children: ReactNode;
}

export const DataCacheProvider: React.FC<DataCacheProviderProps> = ({ children }) => {
  const [data, setData] = useState<CachedData>({
    ganttPriorityView: [],
    ganttResourceView: [],
    detailedSchedule: [],
    scheduleOverview: null,
    isLoading: false,
    error: null,
    lastRefresh: new Date(),
  });

  // Check if we have cached data in localStorage that's still valid
  const [hasValidCache, setHasValidCache] = useState(() => {
    try {
      const savedData = localStorage.getItem('aiOptimizerCache');
      return savedData ? JSON.parse(savedData).ganttPriorityView?.length > 0 : false;
    } catch {
      return false;
    }
  });

  // Always use PPO backend

  // Save data to localStorage
  const saveDataToLocalStorage = (dataToSave: CachedData) => {
    try {
      localStorage.setItem('aiOptimizerCache', JSON.stringify(dataToSave));
    } catch (error) {
      console.warn('Failed to save data to localStorage:', error);
    }
  };

  // Load data from localStorage
  const loadDataFromLocalStorage = (): CachedData | null => {
    try {
      const savedData = localStorage.getItem('aiOptimizerCache');
      if (savedData) {
        const parsedData = JSON.parse(savedData);
        return {
          ...parsedData,
          lastRefresh: new Date(parsedData.lastRefresh),
        };
      }
    } catch (error) {
      console.warn('Failed to load data from localStorage:', error);
    }
    return null;
  };

  const refreshData = async () => {
    console.log('🔄 DataCacheContext: Starting refreshData...');
    setData(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Use PPO backend
      console.log('🤖 DataCacheContext: Using PPO backend');
      
      // First check health
      try {
        const health = await ppoApi.getHealth();
        console.log('📡 PPO Health:', health);
      } catch (healthError) {
        console.warn('Health check failed, continuing anyway:', healthError);
      }
      
      // Create schedule using PPO - load jobs from database
      console.log('🚀 Creating schedule with PPO...');
      const scheduleResponse = await ppoApi.scheduleFromDatabase();
      
      // PPO backend already includes job data in the schedule response
      // No need to fetch separately - use the data directly from the response
      console.log('📊 Using job data from PPO schedule response...');
      console.log(`Total scheduled tasks: ${scheduleResponse.scheduled_jobs.length}`);
      
      // For PPO, we don't have separate job metadata, so create minimal job entries
      // Note: The same job ID can appear multiple times (different operations/machines)
      // We'll create a unique job entry for each unique job ID for metadata purposes
      const uniqueJobIds = [...new Set(scheduleResponse.scheduled_jobs.map(sj => sj.job_id))];
      console.log(`Unique job IDs: ${uniqueJobIds.length}`);
      
      const jobs: Job[] = uniqueJobIds.map(jobId => ({
        job_id: jobId,
        family_id: jobId.split('_')[0] || jobId,
        sequence: 1, // Default sequence, real value would come from database
        processing_time: 4.0, // Default, actual times are in scheduled jobs
        machine_types: [1, 2, 3, 4], // Default, real value from database
        priority: 2, // Default priority
        is_important: false,
        lcd_date: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(), // Default LCD
        setup_time: 0.5,
      }));
      
      // Transform PPO response to match expected formats
      const ganttPriorityData = transformToGanttPriorityView(scheduleResponse.scheduled_jobs, jobs);
      const ganttResourceData = transformToGanttResourceView(scheduleResponse.scheduled_jobs, jobs);
      const detailedScheduleData = transformToDetailedSchedule(scheduleResponse.scheduled_jobs, jobs);
      const scheduleOverviewData = createScheduleOverview(scheduleResponse, jobs);
      
      console.log('📈 PPO DataCacheContext: Data sizes:', {
        ganttPriority: ganttPriorityData.length,
        ganttResource: ganttResourceData.length,
        detailedSchedule: detailedScheduleData.length,
        scheduleOverview: scheduleOverviewData ? 'present' : 'missing'
      });
      
      // Debug: Log sample of transformed data to check Task format
      if (ganttPriorityData.length > 0) {
        console.log('🔍 Sample Gantt Priority Data (first 3 items):', 
          ganttPriorityData.slice(0, 3).map(item => ({
            Task: item.Task,
            JobFamily: item.JobFamily,
            Resource: item.Resource
          }))
        );
      }
      
      const newData: CachedData = {
        ganttPriorityView: ganttPriorityData,
        ganttResourceView: ganttResourceData,
        detailedSchedule: detailedScheduleData,
        scheduleOverview: scheduleOverviewData,
        isLoading: false,
        error: null,
        lastRefresh: new Date(),
      };
      
      setData(newData);
      saveDataToLocalStorage(newData);
      setHasValidCache(true);
      
      console.log('✅ PPO DataCacheContext: Successfully refreshed all data!');
      
    } catch (error) {
      console.error('❌ DataCacheContext: Error during refresh:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch data';
      setData(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  };

  const clearError = () => {
    setData(prev => ({ ...prev, error: null }));
  };

  const clearCache = () => {
    console.log('🗑️ DataCacheContext: Clearing localStorage cache...');
    try {
      localStorage.removeItem('aiOptimizerCache');
      setData({
        ganttPriorityView: [],
        ganttResourceView: [],
        detailedSchedule: [],
        scheduleOverview: null,
        isLoading: false,
        error: null,
        lastRefresh: new Date(),
      });
      setHasValidCache(false);
      console.log('✅ DataCacheContext: Cache cleared successfully');
    } catch (error) {
      console.warn('Failed to clear cache:', error);
    }
  };

  // Load cached data on mount
  useEffect(() => {
    const cachedData = loadDataFromLocalStorage();
    if (cachedData && cachedData.ganttPriorityView.length > 0) {
      setData(cachedData);
      setHasValidCache(true);
    }
  }, []);


  // Provide the context value
  const contextValue: DataCacheContextType = {
    data,
    refreshData,
    clearError,
    clearCache,
  };

  return (
    <DataCacheContext.Provider value={contextValue}>
      {children}
    </DataCacheContext.Provider>
  );
};