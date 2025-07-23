import React, { createContext, useContext, useState, ReactNode } from 'react';
import { ScheduleResponse, HealthResponse } from '../types';

interface ScheduleContextType {
  schedule: ScheduleResponse | null;
  setSchedule: (schedule: ScheduleResponse | null) => void;
  health: HealthResponse | null;
  setHealth: (health: HealthResponse | null) => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
}

const ScheduleContext = createContext<ScheduleContextType | undefined>(undefined);

export const useSchedule = () => {
  const context = useContext(ScheduleContext);
  if (!context) {
    throw new Error('useSchedule must be used within a ScheduleProvider');
  }
  return context;
};

interface ScheduleProviderProps {
  children: ReactNode;
}

export const ScheduleProvider: React.FC<ScheduleProviderProps> = ({ children }) => {
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <ScheduleContext.Provider
      value={{
        schedule,
        setSchedule,
        health,
        setHealth,
        loading,
        setLoading,
        error,
        setError,
      }}
    >
      {children}
    </ScheduleContext.Provider>
  );
};

export default ScheduleContext;