import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Box, Typography, Chip } from '@mui/material';
import { ScheduleResponse, GanttData } from '../types';

interface GanttChartProps {
  schedule: ScheduleResponse;
}

const GanttChart: React.FC<GanttChartProps> = ({ schedule }) => {
  const { chartData, machineNames, maxTime, colors } = useMemo(() => {
    // Get unique machines
    const machines = [...new Set(schedule.scheduled_jobs.map(job => job.machine_name))].sort();
    
    // Create color palette
    const colorPalette = [
      '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1',
      '#d084d0', '#ffb347', '#67b7dc', '#a4de6c', '#ffd93d'
    ];
    
    // Process jobs into Gantt data
    const data: GanttData[] = schedule.scheduled_jobs.map((job, index) => ({
      jobId: job.job_id,
      machineName: job.machine_name,
      startTime: job.start_time,
      endTime: job.end_time,
      duration: job.end_time - job.start_time,
      color: colorPalette[index % colorPalette.length],
    }));
    
    // Group by machine for display
    const machineData = machines.map(machine => {
      const jobs = data
        .filter(job => job.machineName === machine)
        .sort((a, b) => a.startTime - b.startTime);
      
      return { machine, jobs };
    });
    
    const maxEndTime = Math.max(...data.map(d => d.endTime));
    
    return {
      chartData: machineData,
      machineNames: machines,
      maxTime: maxEndTime,
      colors: colorPalette,
    };
  }, [schedule]);

  // Custom bar component for Gantt
  const GanttBar = (props: any) => {
    const { x, y, width, height, job } = props;
    
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={job.color}
          stroke="#333"
          strokeWidth={1}
          rx={2}
          ry={2}
        />
        {width > 30 && (
          <text
            x={x + width / 2}
            y={y + height / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={10}
            fill="#fff"
            fontWeight="bold"
          >
            {job.jobId.slice(-4)}
          </text>
        )}
      </g>
    );
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const job = payload[0].payload;
      return (
        <Box sx={{ bgcolor: 'background.paper', p: 1, border: 1, borderColor: 'divider' }}>
          <Typography variant="body2" fontWeight="bold">
            {job.jobId}
          </Typography>
          <Typography variant="caption" display="block">
            Machine: {job.machineName}
          </Typography>
          <Typography variant="caption" display="block">
            Start: {job.startTime.toFixed(1)}h
          </Typography>
          <Typography variant="caption" display="block">
            End: {job.endTime.toFixed(1)}h
          </Typography>
          <Typography variant="caption" display="block">
            Duration: {job.duration.toFixed(1)}h
          </Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Box>
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Makespan: {schedule.metrics.makespan.toFixed(1)} hours
        </Typography>
        <Chip
          label={`${schedule.metrics.scheduled_jobs}/${schedule.metrics.total_jobs} jobs scheduled`}
          size="small"
          color="primary"
        />
      </Box>

      <Box sx={{ width: '100%', height: 400, overflowX: 'auto' }}>
        {chartData.map((machineData, machineIndex) => (
          <Box key={machineData.machine} sx={{ mb: 2 }}>
            <Typography variant="body2" fontWeight="bold" sx={{ mb: 1 }}>
              {machineData.machine}
            </Typography>
            <Box sx={{ position: 'relative', height: 40, bgcolor: 'grey.100', borderRadius: 1 }}>
              {machineData.jobs.map((job, jobIndex) => {
                const leftPercent = (job.startTime / maxTime) * 100;
                const widthPercent = (job.duration / maxTime) * 100;
                
                return (
                  <Box
                    key={`${job.jobId}-${jobIndex}`}
                    sx={{
                      position: 'absolute',
                      left: `${leftPercent}%`,
                      width: `${widthPercent}%`,
                      height: '100%',
                      bgcolor: job.color,
                      border: 1,
                      borderColor: 'grey.400',
                      borderRadius: 0.5,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      '&:hover': {
                        opacity: 0.8,
                        zIndex: 10,
                      },
                    }}
                    title={`${job.jobId}: ${job.startTime.toFixed(1)}h - ${job.endTime.toFixed(1)}h`}
                  >
                    {widthPercent > 10 && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: 'white',
                          fontWeight: 'bold',
                          fontSize: '0.7rem',
                        }}
                      >
                        {job.jobId.slice(-4)}
                      </Typography>
                    )}
                  </Box>
                );
              })}
            </Box>
          </Box>
        ))}
      </Box>

      {/* Time axis */}
      <Box sx={{ position: 'relative', height: 30, mt: 1 }}>
        <Box sx={{ position: 'absolute', left: 0, top: 0 }}>
          <Typography variant="caption">0h</Typography>
        </Box>
        <Box sx={{ position: 'absolute', left: '50%', top: 0, transform: 'translateX(-50%)' }}>
          <Typography variant="caption">{(maxTime / 2).toFixed(1)}h</Typography>
        </Box>
        <Box sx={{ position: 'absolute', right: 0, top: 0 }}>
          <Typography variant="caption">{maxTime.toFixed(1)}h</Typography>
        </Box>
      </Box>

      {/* Makespan line */}
      <Box
        sx={{
          position: 'relative',
          mt: 2,
          pt: 2,
          borderTop: 2,
          borderColor: 'error.main',
          borderStyle: 'dashed',
        }}
      >
        <Typography variant="caption" color="error" sx={{ position: 'absolute', top: -10, right: 0 }}>
          Makespan: {schedule.metrics.makespan.toFixed(1)}h
        </Typography>
      </Box>
    </Box>
  );
};

export default GanttChart;