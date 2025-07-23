import React, { useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  CircularProgress,
} from '@mui/material';
import HealthStatus from './HealthStatus';
import ScheduleForm from './ScheduleForm';
import GanttChart from './GanttChart';
import MetricsDisplay from './MetricsDisplay';
import api from '../services/api';
import { Job } from '../types';
import { useSchedule } from '../context/ScheduleContext';

const Dashboard: React.FC = () => {
  const { health, setHealth, schedule, setSchedule, loading, setLoading, error, setError } = useSchedule();

  // Check health on mount
  useEffect(() => {
    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const healthData = await api.getHealth();
      setHealth(healthData);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to API');
      setHealth(null);
    }
  };

  const handleSchedule = async (jobs: Job[]) => {
    setLoading(true);
    setError(null);
    
    try {
      const scheduleData = await api.createSchedule(jobs);
      setSchedule(scheduleData);
    } catch (err: any) {
      setError(err.message || 'Failed to create schedule');
      
      // If API fails, create mock data for demonstration
      if (err.message.includes('500') || err.message.includes('connect')) {
        const mockSchedule = createMockSchedule(jobs);
        setSchedule(mockSchedule);
        setError('Using mock data - database not connected');
      }
    } finally {
      setLoading(false);
    }
  };

  const createMockSchedule = (jobs: Job[]): ScheduleResponse => {
    const machines = ['CM03', 'CL02', 'AD02-50HP', 'PP33-250T', 'OV01'];
    const scheduledJobs = [];
    const machineTimes: { [key: string]: number } = {};
    
    // Initialize machine times
    machines.forEach(m => machineTimes[m] = 0);
    
    // Schedule jobs using simple round-robin
    for (let i = 0; i < Math.min(jobs.length, 15); i++) {
      const job = jobs[i];
      const machine = machines[i % machines.length];
      const start = machineTimes[machine];
      const duration = job.processing_time + job.setup_time;
      const end = start + duration;
      
      scheduledJobs.push({
        job_id: job.job_id,
        machine_id: i % machines.length,
        machine_name: machine,
        start_time: start,
        end_time: end,
        start_datetime: new Date(Date.now() + start * 3600000).toISOString(),
        end_datetime: new Date(Date.now() + end * 3600000).toISOString(),
        setup_time_included: job.setup_time,
      });
      
      machineTimes[machine] = end;
    }
    
    const makespan = Math.max(...Object.values(machineTimes));
    
    return {
      schedule_id: 'mock-' + Date.now(),
      scheduled_jobs: scheduledJobs,
      metrics: {
        makespan,
        total_jobs: jobs.length,
        scheduled_jobs: scheduledJobs.length,
        completion_rate: (scheduledJobs.length / jobs.length) * 100,
        average_utilization: 75,
        total_setup_time: scheduledJobs.reduce((sum, j) => sum + j.setup_time_included, 0),
        important_jobs_on_time: 85,
      },
      timestamp: new Date().toISOString(),
    };
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Health Status */}
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2 }}>
              <HealthStatus health={health} onRefresh={checkHealth} />
            </Paper>
          </Grid>

          {/* Schedule Form */}
          <Grid item xs={12} md={9}>
            <Paper sx={{ p: 2 }}>
              <ScheduleForm 
                onSchedule={handleSchedule} 
                disabled={loading || (health !== null && !health.model_loaded)}
              />
            </Paper>
          </Grid>

          {/* Error/Loading Display */}
          {error && (
            <Grid item xs={12}>
              <Alert severity="warning" onClose={() => setError(null)}>
                {error}
              </Alert>
            </Grid>
          )}

          {loading && (
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
                <Typography sx={{ ml: 2 }}>Creating schedule...</Typography>
              </Box>
            </Grid>
          )}

          {/* Metrics Display */}
          {schedule && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <MetricsDisplay metrics={schedule.metrics} />
              </Paper>
            </Grid>
          )}

          {/* Gantt Chart */}
          {schedule && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Schedule Visualization
                </Typography>
                <GanttChart schedule={schedule} />
              </Paper>
            </Grid>
          )}
        </Grid>
      </Container>
  );
};

export default Dashboard;