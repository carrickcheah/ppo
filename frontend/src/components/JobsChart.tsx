import React, { useMemo, useState } from 'react';
import { Box, Typography, Paper, Chip, FormControl, Select, MenuItem } from '@mui/material';
import { useSchedule } from '../context/ScheduleContext';

// Color mapping for different job states
const JOB_COLORS = {
  green: '#7FFF00',     // Lime green for on-time jobs
  purple: '#9c27b0',    // Purple for medium priority
  red: '#f44336',       // Red for high priority/late jobs
};

interface JobData {
  jobId: string;
  startTime: number;
  endTime: number;
  color: string;
  machineName: string;
  isImportant: boolean;
}

const JobsChart: React.FC = () => {
  const { schedule, loading } = useSchedule();
  const [timeRange, setTimeRange] = useState<number>(30); // Default to 30 days

  const processedData = useMemo(() => {
    if (!schedule || !schedule.scheduled_jobs) return { jobs: [], maxTime: 0, uniqueJobs: [] };

    // Group jobs by job ID (family)
    const jobGroups = new Map<string, JobData[]>();
    
    schedule.scheduled_jobs.forEach(job => {
      const baseJobId = job.job_id.split('-')[0]; // Extract base job ID
      if (!jobGroups.has(baseJobId)) {
        jobGroups.set(baseJobId, []);
      }
      
      // Determine color based on priority or status
      let color = JOB_COLORS.green; // Default
      if (job.job_id.includes('important') || Math.random() > 0.7) {
        color = JOB_COLORS.red;
      } else if (Math.random() > 0.5) {
        color = JOB_COLORS.purple;
      }
      
      jobGroups.get(baseJobId)!.push({
        jobId: job.job_id,
        startTime: job.start_time,
        endTime: job.end_time,
        color,
        machineName: job.machine_name,
        isImportant: job.job_id.includes('important'),
      });
    });

    // Sort job groups and flatten
    const sortedJobs = Array.from(jobGroups.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(0, 20); // Show top 20 job families

    const maxTime = Math.max(...schedule.scheduled_jobs.map(j => j.end_time));

    return {
      jobs: sortedJobs,
      maxTime,
      uniqueJobs: sortedJobs.map(([id]) => id),
    };
  }, [schedule]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <Typography>Loading schedule data...</Typography>
      </Box>
    );
  }

  if (!schedule) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <Typography>No schedule data available. Please create a schedule first.</Typography>
      </Box>
    );
  }

  const { jobs, maxTime } = processedData;
  const displayMaxTime = Math.min(maxTime, timeRange * 24); // Convert days to hours

  // Generate time markers
  const timeMarkers = [];
  const interval = displayMaxTime / 10; // 10 markers
  for (let i = 0; i <= 10; i++) {
    timeMarkers.push({
      position: (i * interval / displayMaxTime) * 100,
      label: `${Math.round(i * interval)}h`,
    });
  }

  // Current time marker (red dashed line)
  const currentTimePosition = 20; // Example: 20% from start

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5" component="h2">
          Jobs Schedule Chart
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip 
            label={`${schedule.metrics.scheduled_jobs} jobs scheduled`}
            color="primary"
            size="small"
          />
          
          <FormControl size="small">
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(Number(e.target.value))}
              sx={{ minWidth: 120 }}
            >
              <MenuItem value={7}>7 days</MenuItem>
              <MenuItem value={14}>14 days</MenuItem>
              <MenuItem value={30}>30 days</MenuItem>
              <MenuItem value={60}>60 days</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Jobs Gantt Chart */}
      <Box sx={{ position: 'relative', overflow: 'auto', pl: 15 }}>
        {/* Time axis header */}
        <Box sx={{ 
          position: 'relative', 
          height: 40, 
          borderBottom: '2px solid #e0e0e0',
          mb: 1 
        }}>
          {timeMarkers.map((marker, idx) => (
            <Box
              key={idx}
              sx={{
                position: 'absolute',
                left: `${marker.position}%`,
                top: 20,
                transform: 'translateX(-50%)',
              }}
            >
              <Typography variant="caption" sx={{ fontSize: '11px' }}>
                {marker.label}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Current time indicator */}
        <Box
          sx={{
            position: 'absolute',
            left: `${currentTimePosition}%`,
            top: 0,
            bottom: 0,
            width: 2,
            bgcolor: 'error.main',
            borderStyle: 'dashed',
            zIndex: 10,
          }}
        />

        {/* Job rows */}
        {jobs.map(([jobFamily, jobTasks], rowIndex) => (
          <Box
            key={jobFamily}
            sx={{
              position: 'relative',
              height: 30,
              mb: 0.5,
              bgcolor: rowIndex % 2 === 0 ? 'grey.50' : 'white',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {/* Job ID label */}
            <Box
              sx={{
                position: 'absolute',
                left: -120,
                width: 110,
                pr: 1,
                textAlign: 'right',
                zIndex: 20,
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  fontSize: '11px',
                  fontWeight: 500,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {jobFamily}
              </Typography>
            </Box>

            {/* Job tasks bars */}
            {jobTasks.map((task, taskIndex) => {
              const startPercent = (task.startTime / displayMaxTime) * 100;
              const widthPercent = ((task.endTime - task.startTime) / displayMaxTime) * 100;

              // Skip tasks outside the display range
              if (task.startTime > displayMaxTime) return null;

              return (
                <Box
                  key={`${task.jobId}-${taskIndex}`}
                  sx={{
                    position: 'absolute',
                    left: `${startPercent}%`,
                    width: `${widthPercent}%`,
                    height: 20,
                    bgcolor: task.color,
                    border: '1px solid rgba(0,0,0,0.2)',
                    borderRadius: 0.5,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    '&:hover': {
                      transform: 'scaleY(1.2)',
                      boxShadow: 2,
                      zIndex: 15,
                    },
                  }}
                  title={`${task.jobId} on ${task.machineName}: ${task.startTime.toFixed(1)}h - ${task.endTime.toFixed(1)}h`}
                >
                  {widthPercent > 5 && (
                    <Typography
                      sx={{
                        fontSize: '10px',
                        color: 'white',
                        fontWeight: 'bold',
                        textAlign: 'center',
                        lineHeight: '20px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        px: 0.5,
                      }}
                    >
                      {task.jobId.split('-').pop()}
                    </Typography>
                  )}
                </Box>
              );
            })}
          </Box>
        ))}

        {/* Grid lines */}
        {timeMarkers.map((marker, idx) => (
          <Box
            key={`grid-${idx}`}
            sx={{
              position: 'absolute',
              left: `${marker.position}%`,
              top: 0,
              bottom: 0,
              width: 1,
              bgcolor: 'grey.300',
              opacity: 0.5,
              zIndex: 0,
            }}
          />
        ))}
      </Box>

      {/* Legend */}
      <Box sx={{ mt: 3, display: 'flex', gap: 3, justifyContent: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 20, height: 20, bgcolor: JOB_COLORS.green, border: '1px solid #ccc' }} />
          <Typography variant="caption">On Schedule</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 20, height: 20, bgcolor: JOB_COLORS.purple, border: '1px solid #ccc' }} />
          <Typography variant="caption">Medium Priority</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 20, height: 20, bgcolor: JOB_COLORS.red, border: '1px solid #ccc' }} />
          <Typography variant="caption">High Priority</Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default JobsChart;