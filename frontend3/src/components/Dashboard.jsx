import React from 'react';
import JobsGanttChart from './JobsGanttChart';
import MachineGanttChart from './MachineGanttChart';

const Dashboard = ({ 
  scheduleData, 
  loading, 
  error, 
  selectedDataset,
  setSelectedDataset,
  selectedModel,
  setSelectedModel,
  timeRange,
  setTimeRange,
  handleSchedule,
  activeChart,
  setActiveChart,
  isFullscreen,
  setIsFullscreen
}) => {
  return (
    <div className="dashboard">
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
        
        <div className="control-group">
          <label htmlFor="time-range">Time Range:</label>
          <select 
            id="time-range"
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="2days">2 Days</option>
            <option value="2weeks">2 Weeks</option>
            <option value="4weeks">4 Weeks</option>
            <option value="6weeks">6 Weeks</option>
          </select>
        </div>
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
            <button 
              className="fullscreen-button"
              onClick={() => setIsFullscreen(!isFullscreen)}
              title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
            >
              {isFullscreen ? '⊖ Exit Fullscreen' : '⊕ Fullscreen'}
            </button>
          </div>

          <div className={`chart-container ${isFullscreen ? 'fullscreen' : ''}`}>
            {isFullscreen && (
              <button 
                className="close-fullscreen"
                onClick={() => setIsFullscreen(false)}
              >
                ✕ Close
              </button>
            )}
            {activeChart === 'jobs' ? (
              <JobsGanttChart jobs={scheduleData.jobs} timeRange={timeRange} />
            ) : (
              <MachineGanttChart machines={scheduleData.machines} timeRange={timeRange} />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;