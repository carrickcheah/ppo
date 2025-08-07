import React from 'react';
import JobsGanttChart from './JobsGanttChart';

const JobsFullPage = ({ scheduleData, timeRange, setTimeRange }) => {
  if (!scheduleData) {
    return (
      <div className="jobs-page">
        <div className="jobs-empty-state">
          <div className="empty-icon">
            <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="#e0e0e0" strokeWidth="1">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <line x1="9" y1="9" x2="15" y2="9"/>
              <line x1="9" y1="12" x2="15" y2="12"/>
              <line x1="9" y1="15" x2="15" y2="15"/>
            </svg>
          </div>
          <h2>No Schedule Data Available</h2>
          <p>Navigate to the Dashboard to schedule production jobs</p>
          <a href="/" className="action-button primary">Go to Dashboard</a>
        </div>
      </div>
    );
  }

  const scheduledCount = scheduleData.jobs?.length || 0;
  const onTimeCount = scheduleData.jobs?.filter(j => j.days_to_deadline > 0).length || 0;
  const completionRate = scheduleData.statistics?.completion_rate || 0;

  return (
    <div className="jobs-page">
      <div className="jobs-header">
        <div className="header-content">
          <div className="header-left">
            <h1>Production Jobs Allocation</h1>
            <p className="header-subtitle">Real-time scheduling visualization powered by PPO optimization</p>
          </div>
          <div className="header-right">
            <div className="header-stats">
              <div className="stat-badge">
                <span className="stat-value">{scheduledCount}</span>
                <span className="stat-label">Jobs</span>
              </div>
              <div className="stat-badge">
                <span className="stat-value">{completionRate.toFixed(0)}%</span>
                <span className="stat-label">Completion</span>
              </div>
              <div className="stat-badge">
                <span className="stat-value">{onTimeCount}</span>
                <span className="stat-label">On Time</span>
              </div>
            </div>
          </div>
        </div>
        <div className="header-controls">
          <div className="control-section">
            <label className="control-label">Time Range</label>
            <div className="time-range-buttons">
              <button 
                className={`range-btn ${timeRange === '2days' ? 'active' : ''}`}
                onClick={() => setTimeRange('2days')}
              >
                2D
              </button>
              <button 
                className={`range-btn ${timeRange === '5days' ? 'active' : ''}`}
                onClick={() => setTimeRange('5days')}
              >
                5D
              </button>
              <button 
                className={`range-btn ${timeRange === '2weeks' ? 'active' : ''}`}
                onClick={() => setTimeRange('2weeks')}
              >
                2W
              </button>
              <button 
                className={`range-btn ${timeRange === '4weeks' ? 'active' : ''}`}
                onClick={() => setTimeRange('4weeks')}
              >
                4W
              </button>
              <button 
                className={`range-btn ${timeRange === '6weeks' ? 'active' : ''}`}
                onClick={() => setTimeRange('6weeks')}
              >
                6W
              </button>
            </div>
          </div>
          <div className="control-section">
            <button className="action-button secondary" onClick={() => window.location.reload()}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
              </svg>
              Refresh
            </button>
          </div>
        </div>
      </div>
      <div className="jobs-chart-container">
        <div className="chart-wrapper">
          <JobsGanttChart jobs={scheduleData.jobs} timeRange={timeRange} />
        </div>
      </div>
    </div>
  );
};

export default JobsFullPage;