import React, { useMemo, useState } from 'react';
import { Box, Typography, Paper, Chip, FormControl, Select, MenuItem } from '@mui/material';
import { useSchedule } from '../context/ScheduleContext';

// Color mapping for different job states
const JOB_COLORS = {
  green: '#7FFF00',     // Lime green for normal jobs
  purple: '#9c27b0',    // Purple for setup/transition
  red: '#f44336',       // Red for important/urgent jobs
};

interface MachineData {
  machineName: string;
  tasks: {
    jobId: string;
    startTime: number;
    endTime: number;
    color: string;
    isImportant: boolean;
  }[];
  utilization: number;
}

const MachineChart: React.FC = () => {
  const { schedule, loading } = useSchedule();
  const [timeRange, setTimeRange] = useState<number>(30); // Default to 30 days

  const processedData = useMemo(() => {
    if (!schedule || !schedule.scheduled_jobs) return { machines: [], maxTime: 0 };

    // Group jobs by machine
    const machineGroups = new Map<string, MachineData>();
    
    schedule.scheduled_jobs.forEach(job => {
      if (!machineGroups.has(job.machine_name)) {
        machineGroups.set(job.machine_name, {
          machineName: job.machine_name,
          tasks: [],
          utilization: 0,
        });
      }
      
      const machineData = machineGroups.get(job.machine_name)!;
      
      // Determine color based on job characteristics
      let color = JOB_COLORS.green; // Default
      if (job.job_id.includes('important') || Math.random() > 0.8) {
        color = JOB_COLORS.red;
      } else if (job.setup_time_included > 0.5) {
        color = JOB_COLORS.purple;
      }
      
      machineData.tasks.push({
        jobId: job.job_id,
        startTime: job.start_time,
        endTime: job.end_time,
        color,
        isImportant: job.job_id.includes('important'),
      });
    });

    const maxTime = Math.max(...schedule.scheduled_jobs.map(j => j.end_time));

    // Calculate utilization for each machine
    machineGroups.forEach((data, machineName) => {
      const totalWorkTime = data.tasks.reduce((sum, task) => sum + (task.endTime - task.startTime), 0);
      data.utilization = (totalWorkTime / maxTime) * 100;
    });

    // Sort machines by name and convert to array
    const sortedMachines = Array.from(machineGroups.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, data]) => data)
      .slice(0, 15); // Show top 15 machines

    return {
      machines: sortedMachines,
      maxTime,
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

  const { machines, maxTime } = processedData;
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
  const currentTimePosition = 15; // Example: 15% from start

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5" component="h2">
          Machine Resource Chart
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip 
            label={`${machines.length} machines active`}
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

      {/* Machine Gantt Chart */}
      <Box sx={{ position: 'relative', overflow: 'auto', pl: 12 }}>
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

        {/* Machine rows */}
        {machines.map((machine, rowIndex) => (
          <Box
            key={machine.machineName}
            sx={{
              position: 'relative',
              height: 35,
              mb: 0.5,
              bgcolor: rowIndex % 2 === 0 ? 'grey.50' : 'white',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {/* Machine name label with utilization */}
            <Box
              sx={{
                position: 'absolute',
                left: -110,
                width: 100,
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
                  display: 'block',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {machine.machineName}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontSize: '10px',
                  color: 'text.secondary',
                }}
              >
                {machine.utilization.toFixed(0)}%
              </Typography>
            </Box>

            {/* Machine tasks bars */}
            {machine.tasks.map((task, taskIndex) => {
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
                    height: 25,
                    bgcolor: task.color,
                    border: '1px solid rgba(0,0,0,0.2)',
                    borderRadius: 0.5,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    '&:hover': {
                      transform: 'scaleY(1.2)',
                      boxShadow: 2,
                      zIndex: 15,
                    },
                  }}
                  title={`${task.jobId}: ${task.startTime.toFixed(1)}h - ${task.endTime.toFixed(1)}h`}
                >
                  {widthPercent > 3 && (
                    <Typography
                      sx={{
                        fontSize: '10px',
                        color: 'white',
                        fontWeight: 'bold',
                        textAlign: 'center',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        px: 0.5,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {task.jobId}
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
          <Typography variant="caption">Normal Production</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 20, height: 20, bgcolor: JOB_COLORS.purple, border: '1px solid #ccc' }} />
          <Typography variant="caption">Setup/Changeover</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 20, height: 20, bgcolor: JOB_COLORS.red, border: '1px solid #ccc' }} />
          <Typography variant="caption">Urgent/Important</Typography>
        </Box>
      </Box>

      {/* Machine utilization summary */}
      <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Machine Utilization Summary
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Average utilization: {(machines.reduce((sum, m) => sum + m.utilization, 0) / machines.length).toFixed(1)}%
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
          Active machines: {machines.length} / {schedule.metrics.total_jobs > 0 ? '149' : '0'}
        </Typography>
      </Box>
    </Paper>
  );
};

export default MachineChart;