import React from 'react';

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
  models
}) => {
  return (
    <div className="dashboard">
      <div className="control-panel">
        <div className="control-group">
          <label htmlFor="dataset-select">Dataset</label>
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
            <option value="80_jobs">80 Jobs (260 tasks)</option>
            <option value="100_jobs">100 Jobs (327 tasks)</option>
            <option value="150_jobs">150 Jobs (490 tasks)</option>
            <option value="180_jobs">180 Jobs (588 tasks)</option>
            <option value="200_jobs">200 Jobs (653 tasks)</option>
            <option value="250_jobs">250 Jobs (816 tasks)</option>
            <option value="300_jobs">300 Jobs (980 tasks)</option>
            <option value="330_jobs">330 Jobs (1078 tasks)</option>
            <option value="380_jobs">380 Jobs (1241 tasks)</option>
            <option value="400_jobs">400 Jobs (1306 tasks)</option>
            <option value="430_jobs">430 Jobs (1404 tasks)</option>
            <option value="450_jobs">450 Jobs (1469 tasks)</option>
            <option value="500_jobs">500 Jobs (1633 tasks)</option>
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="model-select">Model</label>
          <select 
            id="model-select"
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={loading}
          >
            {models && models.map(model => (
              <option key={model.name} value={model.name}>
                {model.name.replace(/_/g, ' ').toUpperCase()} 
                {model.training_steps && ` (${model.training_steps.toLocaleString()} steps)`}
              </option>
            ))}
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
          <label htmlFor="time-range">Time Range</label>
          <select 
            id="time-range"
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="5days">5 Days</option>
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
            <div className="stats-grid-2x4">
              <div className="stat-item">
                <span className="stat-label">Completion Rate</span>
                <span className="stat-value">
                  {scheduleData.statistics?.completion_rate?.toFixed(1)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Scheduled Tasks</span>
                <span className="stat-value">
                  {scheduleData.statistics?.scheduled_tasks} / {scheduleData.statistics?.total_tasks}
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">On-Time Rate</span>
                <span className="stat-value">
                  {scheduleData.statistics?.on_time_rate?.toFixed(1)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Machine Utilization</span>
                <span className="stat-value">
                  {scheduleData.statistics?.machine_utilization?.toFixed(1)}%
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Makespan</span>
                <span className="stat-value">
                  {scheduleData.statistics?.makespan?.toFixed(0)} hours
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Inference Time</span>
                <span className="stat-value">
                  {scheduleData.statistics?.inference_time?.toFixed(2)} seconds
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Total Jobs</span>
                <span className="stat-value">
                  {scheduleData.jobs?.length || 0}
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Late Jobs</span>
                <span className="stat-value">
                  {scheduleData.jobs?.filter(j => j.days_to_deadline < 0).length || 0}
                </span>
              </div>
            </div>
          </div>

        </div>
      )}
    </div>
  );
};

export default Dashboard;