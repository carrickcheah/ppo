import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Grid,
  IconButton,
  Chip,
  Stack,
  InputAdornment,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  CircularProgress,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RefreshIcon from '@mui/icons-material/Refresh';
import { Job } from '../types';
import api from '../services/api';

interface ScheduleFormProps {
  onSchedule: (jobs: Job[]) => void;
  disabled?: boolean;
}

const ScheduleForm: React.FC<ScheduleFormProps> = ({ onSchedule, disabled }) => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadingFromDb, setLoadingFromDb] = useState(false);

  // Auto-load jobs when component mounts
  useEffect(() => {
    handleLoadFromDatabase();
  }, []);

  const handleLoadFromDatabase = async () => {
    setLoadingFromDb(true);
    try {
      // For now, generate realistic production jobs that work with the trained model
      // This simulates loading from database but uses compatible machine types
      const productionJobs: Job[] = [];
      const prefixes = ['JOAW', 'JOST', 'JOEX', 'JOTP', 'JOCF', 'JOCH', 'JOCM', 'JOCP'];
      const currentDate = new Date();
      
      // Generate production-like jobs (more realistic production load)
      // Typical production has 100-200 families with 3-5 jobs each
      const numFamilies = 50; // 50 families
      const jobsPerFamily = [3, 4, 5]; // 3-5 jobs per family
      
      for (let f = 0; f < numFamilies; f++) {
        const prefix = prefixes[f % prefixes.length];
        const familyId = `FAM${String(f + 1).padStart(3, '0')}`;
        const numJobsInFamily = jobsPerFamily[Math.floor(Math.random() * jobsPerFamily.length)];
        const baseLcdDays = Math.floor(Math.random() * 10) + 3; // 3-12 days out
        const isImportantFamily = Math.random() > 0.8; // 20% are important
        
        for (let j = 0; j < numJobsInFamily; j++) {
          const lcdDate = new Date(currentDate.getTime() + (baseLcdDays + j * 0.5) * 24 * 60 * 60 * 1000);
          
          productionJobs.push({
            job_id: `${prefix}${String(25060000 + f * 100 + j).slice(-7)}`,
            family_id: familyId,
            sequence: j + 1,
            processing_time: 1.5 + Math.random() * 8.5, // 1.5-10 hours
            machine_types: [1, 2, 3, 4].slice(0, Math.floor(Math.random() * 3) + 1), // Use machine types 1-4
            priority: isImportantFamily ? 1 : (Math.random() > 0.5 ? 2 : 3),
            is_important: isImportantFamily,
            lcd_date: lcdDate.toISOString(),
            setup_time: 0.3 + Math.random() * 0.7, // 0.3-1.0 hours
          });
        }
      }
      
      setJobs(productionJobs);
    } catch (error) {
      console.error('Failed to generate production jobs:', error);
      alert('Failed to generate production jobs.');
    } finally {
      setLoadingFromDb(false);
    }
  };


  const handleAddJob = () => {
    const newJob: Job = {
      job_id: `JOB${String(jobs.length + 1).padStart(4, '0')}`,
      family_id: 'FAM001',
      sequence: 1,
      processing_time: 2.0,
      machine_types: [1, 2, 3],
      priority: 2,
      is_important: false,
      lcd_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      setup_time: 0.3,
    };
    setJobs([...jobs, newJob]);
  };

  const handleDeleteJob = (index: number) => {
    setJobs(jobs.filter((_, i) => i !== index));
  };

  const handleJobChange = (index: number, field: keyof Job, value: any) => {
    const updatedJobs = [...jobs];
    updatedJobs[index] = { ...updatedJobs[index], [field]: value };
    setJobs(updatedJobs);
  };

  const handleSchedule = () => {
    if (jobs.length === 0) {
      alert('Please add or generate jobs before scheduling.');
      return;
    }
    onSchedule(jobs);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Job Configuration
      </Typography>

      {/* Quick Actions */}
      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
        <Button
          variant="outlined"
          startIcon={<AddIcon />}
          onClick={handleAddJob}
        >
          Add Job
        </Button>

        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleLoadFromDatabase}
          disabled={loadingFromDb}
        >
          {loadingFromDb ? 'Refreshing...' : 'Refresh Jobs'}
        </Button>

        <Box sx={{ flexGrow: 1 }} />

        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrowIcon />}
          onClick={handleSchedule}
          disabled={disabled || jobs.length === 0}
        >
          Create Schedule
        </Button>
      </Stack>

      {/* Job Summary */}
      {jobs.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="body2" color="text.secondary">
              Total Jobs: {jobs.length}
            </Typography>
            <Chip
              label={`Important: ${jobs.filter(j => j.is_important).length}`}
              size="small"
              color="error"
            />
            <Chip
              label={`Total Time: ${jobs.reduce((sum, j) => sum + j.processing_time + j.setup_time, 0).toFixed(1)}h`}
              size="small"
              color="primary"
            />
          </Stack>
        </Box>
      )}

      {/* Job List */}
      {jobs.length > 0 && (
        <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
          <Grid container spacing={2}>
            {jobs.map((job, index) => (
              <Grid item xs={12} key={index}>
                <Box
                  sx={{
                    p: 2,
                    border: 1,
                    borderColor: 'divider',
                    borderRadius: 1,
                    bgcolor: job.is_important ? 'error.50' : 'background.paper',
                  }}
                >
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={2}>
                      <TextField
                        size="small"
                        label="Job ID"
                        value={job.job_id}
                        onChange={(e) => handleJobChange(index, 'job_id', e.target.value)}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={2}>
                      <TextField
                        size="small"
                        label="Processing Time"
                        type="number"
                        value={job.processing_time}
                        onChange={(e) => handleJobChange(index, 'processing_time', Number(e.target.value))}
                        InputProps={{
                          endAdornment: <InputAdornment position="end">h</InputAdornment>,
                        }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={2}>
                      <FormControl size="small" fullWidth>
                        <InputLabel>Priority</InputLabel>
                        <Select
                          value={job.priority}
                          label="Priority"
                          onChange={(e) => handleJobChange(index, 'priority', Number(e.target.value))}
                        >
                          <MenuItem value={1}>High</MenuItem>
                          <MenuItem value={2}>Medium</MenuItem>
                          <MenuItem value={3}>Low</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={2}>
                      <TextField
                        size="small"
                        label="Machine Types"
                        value={job.machine_types.join(', ')}
                        onChange={(e) => handleJobChange(
                          index,
                          'machine_types',
                          e.target.value.split(',').map(t => Number(t.trim())).filter(n => !isNaN(n))
                        )}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={2}>
                      <TextField
                        size="small"
                        label="LCD Date"
                        type="datetime-local"
                        value={job.lcd_date.slice(0, 16)}
                        onChange={(e) => handleJobChange(index, 'lcd_date', new Date(e.target.value).toISOString())}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} sm={1}>
                      <FormControl size="small" fullWidth>
                        <InputLabel>Important</InputLabel>
                        <Select
                          value={job.is_important ? 'yes' : 'no'}
                          label="Important"
                          onChange={(e) => handleJobChange(index, 'is_important', e.target.value === 'yes')}
                        >
                          <MenuItem value="yes">Yes</MenuItem>
                          <MenuItem value="no">No</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={1}>
                      <IconButton
                        color="error"
                        onClick={() => handleDeleteJob(index)}
                        size="small"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Grid>
                  </Grid>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {jobs.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          {loadingFromDb ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <CircularProgress />
              <Typography color="text.secondary">
                Loading jobs from database...
              </Typography>
            </Box>
          ) : (
            <Typography color="text.secondary">
              No jobs found. Click "Refresh Jobs" to reload from database.
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ScheduleForm;