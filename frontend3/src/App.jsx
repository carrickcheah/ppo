import React, { useState, useEffect } from 'react';
import './App.css';
import schedulingApi from './services/api';
import JobsGanttChart from './components/JobsGanttChart';
import MachineGanttChart from './components/MachineGanttChart';

function App() {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('100_jobs');
  const [selectedModel, setSelectedModel] = useState('sb3_1million');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scheduleData, setScheduleData] = useState(null);
  const [activeChart, setActiveChart] = useState('jobs');

  // Load datasets and models on mount
  useEffect(() => {
    loadOptions();
  }, []);

  const loadOptions = async () => {
    try {
      const [datasetsRes, modelsRes] = await Promise.all([
        schedulingApi.getDatasets(),
        schedulingApi.getModels()
      ]);
      setDatasets(datasetsRes.datasets || []);
      setModels(modelsRes.models || []);
    } catch (err) {
      console.error('Error loading options:', err);
      setError('Failed to load datasets and models');
    }
  };

  const handleSchedule = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await schedulingApi.scheduleJobs(selectedDataset, selectedModel);
      setScheduleData(result);
    } catch (err) {
      console.error('Scheduling error:', err);
      setError(err.response?.data?.detail || 'Failed to schedule jobs');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>PPO Scheduling Visualization System</h1>
        <p>Schedule jobs using trained PPO models and visualize with Gantt charts</p>
      </header>

      <div className="control-panel">
        <div className="control-group">
          <label htmlFor="dataset-select">Dataset:</label>
          <select 
            id="dataset-select"
            value={selectedDataset} 
            onChange={(e) => setSelectedDataset(e.target.value)}
            disabled={loading}
          >
            <option value="10_jobs">10 Jobs (34 tasks)</option>
            <option value="20_jobs">20 Jobs (65 tasks)</option>
            <option value="40_jobs">40 Jobs (130 tasks)</option>
            <option value="60_jobs">60 Jobs (195 tasks)</option>
            <option value="100_jobs">100 Jobs (327 tasks) ⭐</option>
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="model-select">Model:</label>
          <select 
            id="model-select"
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={loading}
          >
            <option value="sb3_1million">SB3 1M Steps (Works with all datasets) ⭐</option>
            <option value="sb3_500k">SB3 500K Steps (Works with all datasets)</option>
            <option value="sb3_optimized">SB3 Optimized (Works with all datasets)</option>
          </select>
        </div>

        <button 
          className="schedule-button"
          onClick={handleSchedule} 
          disabled={loading}
        >
          {loading ? 'Scheduling...' : 'Schedule Jobs'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {scheduleData && (
        <div className="results-container">
          <div className="statistics-panel">
            <h3>Scheduling Statistics</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Completion Rate:</span>
                <span className="stat-value">
                  {scheduleData.statistics?.completion_rate?.toFixed(1)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Scheduled Tasks:</span>
                <span className="stat-value">
                  {scheduleData.statistics?.scheduled_tasks} / {scheduleData.statistics?.total_tasks}
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">On-Time Rate:</span>
                <span className="stat-value">
                  {scheduleData.statistics?.on_time_rate?.toFixed(1)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Machine Utilization:</span>
                <span className="stat-value">
                  {scheduleData.statistics?.machine_utilization?.toFixed(1)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Makespan:</span>
                <span className="stat-value">
                  {scheduleData.statistics?.makespan?.toFixed(0)} hours
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Inference Time:</span>
                <span className="stat-value">
                  {scheduleData.statistics?.inference_time?.toFixed(2)} seconds
                </span>
              </div>
            </div>
          </div>

          <div className="chart-tabs">
            <button 
              className={`tab-button ${activeChart === 'jobs' ? 'active' : ''}`}
              onClick={() => setActiveChart('jobs')}
            >
              Jobs Allocation Chart
            </button>
            <button 
              className={`tab-button ${activeChart === 'machines' ? 'active' : ''}`}
              onClick={() => setActiveChart('machines')}
            >
              Machine Allocation Chart
            </button>
          </div>

          <div className="chart-container">
            {activeChart === 'jobs' ? (
              <JobsGanttChart jobs={scheduleData.jobs} />
            ) : (
              <MachineGanttChart machines={scheduleData.machines} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App
