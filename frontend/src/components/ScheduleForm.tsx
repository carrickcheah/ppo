import React, { useState } from 'react';
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
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { Job } from '../types';
import api from '../services/api';

interface ScheduleFormProps {
  onSchedule: (jobs: Job[]) => void;
  disabled?: boolean;
}

const ScheduleForm: React.FC<ScheduleFormProps> = ({ onSchedule, disabled }) => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobCount, setJobCount] = useState(20);

  const handleGenerateSampleJobs = () => {
    const sampleJobs = api.generateSampleJobs(jobCount);
    setJobs(sampleJobs);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        const loadedJobs = await api.loadJobsFromFile(file);
        setJobs(loadedJobs);
      } catch (error) {
        console.error('Failed to load jobs from file:', error);
        alert('Failed to load jobs from file. Please check the format.');
      }
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
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TextField
            type="number"
            size="small"
            value={jobCount}
            onChange={(e) => setJobCount(Number(e.target.value))}
            InputProps={{
              inputProps: { min: 1, max: 100 },
              endAdornment: <InputAdornment position="end">jobs</InputAdornment>,
            }}
            sx={{ width: 120 }}
          />
          <Button
            variant="outlined"
            startIcon={<AutoFixHighIcon />}
            onClick={handleGenerateSampleJobs}
          >
            Generate Sample
          </Button>
        </Box>

        <Button
          variant="outlined"
          component="label"
          startIcon={<UploadFileIcon />}
        >
          Load from File
          <input
            type="file"
            accept=".json"
            hidden
            onChange={handleFileUpload}
          />
        </Button>

        <Button
          variant="outlined"
          startIcon={<AddIcon />}
          onClick={handleAddJob}
        >
          Add Job
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
          <Typography color="text.secondary">
            No jobs configured. Generate sample jobs or add manually.
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ScheduleForm;