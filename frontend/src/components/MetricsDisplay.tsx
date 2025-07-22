import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
} from '@mui/material';
import SpeedIcon from '@mui/icons-material/Speed';
import AssignmentTurnedInIcon from '@mui/icons-material/AssignmentTurnedIn';
import TimerIcon from '@mui/icons-material/Timer';
import PrecisionManufacturingIcon from '@mui/icons-material/PrecisionManufacturing';
import BuildIcon from '@mui/icons-material/Build';
import PriorityHighIcon from '@mui/icons-material/PriorityHigh';
import { ScheduleMetrics } from '../types';

interface MetricsDisplayProps {
  metrics: ScheduleMetrics;
}

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
  const metricCards = [
    {
      title: 'Makespan',
      value: `${metrics.makespan.toFixed(1)} hours`,
      icon: <TimerIcon />,
      color: '#1976d2',
      description: 'Total time to complete all jobs',
    },
    {
      title: 'Completion Rate',
      value: `${metrics.completion_rate.toFixed(1)}%`,
      icon: <AssignmentTurnedInIcon />,
      color: metrics.completion_rate === 100 ? '#4caf50' : '#ff9800',
      description: `${metrics.scheduled_jobs} of ${metrics.total_jobs} jobs scheduled`,
      progress: metrics.completion_rate,
    },
    {
      title: 'Machine Utilization',
      value: `${metrics.average_utilization.toFixed(1)}%`,
      icon: <PrecisionManufacturingIcon />,
      color: '#9c27b0',
      description: 'Average machine busy time',
      progress: metrics.average_utilization,
    },
    {
      title: 'Setup Time',
      value: `${metrics.total_setup_time.toFixed(1)} hours`,
      icon: <BuildIcon />,
      color: '#f44336',
      description: 'Total setup time across all jobs',
    },
    {
      title: 'Important Jobs On Time',
      value: `${metrics.important_jobs_on_time.toFixed(0)}%`,
      icon: <PriorityHighIcon />,
      color: metrics.important_jobs_on_time >= 90 ? '#4caf50' : '#ff5722',
      description: 'Critical jobs meeting LCD dates',
      progress: metrics.important_jobs_on_time,
    },
    {
      title: 'Efficiency Score',
      value: `${((metrics.completion_rate + metrics.average_utilization + metrics.important_jobs_on_time) / 3).toFixed(0)}%`,
      icon: <SpeedIcon />,
      color: '#00bcd4',
      description: 'Overall schedule efficiency',
      progress: (metrics.completion_rate + metrics.average_utilization + metrics.important_jobs_on_time) / 3,
    },
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Schedule Performance Metrics
      </Typography>
      
      <Grid container spacing={2}>
        {metricCards.map((metric, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card variant="outlined" sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                  <Box
                    sx={{
                      p: 1,
                      borderRadius: 1,
                      bgcolor: metric.color + '20',
                      color: metric.color,
                      mr: 2,
                    }}
                  >
                    {metric.icon}
                  </Box>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      {metric.title}
                    </Typography>
                    <Typography variant="h5" sx={{ fontWeight: 'bold', color: metric.color }}>
                      {metric.value}
                    </Typography>
                  </Box>
                </Box>
                
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  {metric.description}
                </Typography>
                
                {metric.progress !== undefined && (
                  <LinearProgress
                    variant="determinate"
                    value={metric.progress}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      bgcolor: metric.color + '20',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: metric.color,
                        borderRadius: 3,
                      },
                    }}
                  />
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Summary chips */}
      <Box sx={{ mt: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {metrics.completion_rate === 100 && (
          <Chip
            label="All Jobs Scheduled"
            color="success"
            size="small"
          />
        )}
        {metrics.average_utilization > 80 && (
          <Chip
            label="High Utilization"
            color="primary"
            size="small"
          />
        )}
        {metrics.important_jobs_on_time === 100 && (
          <Chip
            label="All Critical Jobs On Time"
            color="success"
            size="small"
          />
        )}
        {metrics.makespan < 50 && (
          <Chip
            label="Optimal Makespan"
            color="info"
            size="small"
          />
        )}
      </Box>
    </Box>
  );
};

export default MetricsDisplay;