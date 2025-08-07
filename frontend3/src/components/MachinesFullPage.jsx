import React from 'react';
import MachineGanttChart from './MachineGanttChart';

const MachinesFullPage = ({ scheduleData, timeRange, setTimeRange }) => {
  if (!scheduleData) {
    return (
      <div className="machines-page">
        <div className="machines-empty-state">
          <div className="empty-icon">
            <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="#e0e0e0" strokeWidth="1">
              <rect x="3" y="6" width="18" height="12" rx="2" ry="2"/>
              <line x1="8" y1="12" x2="16" y2="12"/>
              <circle cx="6" cy="12" r="2"/>
              <circle cx="18" cy="12" r="2"/>
            </svg>
          </div>
          <h2>No Schedule Data Available</h2>
          <p>Navigate to the Dashboard to schedule production jobs</p>
          <a href="/" className="action-button primary">Go to Dashboard</a>
        </div>
      </div>
    );
  }

  const machineCount = scheduleData.machines?.length || 0;
  const utilization = scheduleData.statistics?.machine_utilization || 0;
  const makespan = scheduleData.statistics?.makespan || 0;

  return (
    <div className="machines-page">
      <div className="machines-header">
        <div className="header-content">
          <div className="header-left">
            <h1>Machine Allocation View</h1>
            <p className="header-subtitle">Real-time machine utilization powered by PPO optimization</p>
          </div>
          <div className="header-right">
            <div className="header-stats">
              <div className="stat-badge">
                <span className="stat-value">{machineCount}</span>
                <span className="stat-label">Machines</span>
              </div>
              <div className="stat-badge">
                <span className="stat-value">{utilization.toFixed(0)}%</span>
                <span className="stat-label">Utilization</span>
              </div>
              <div className="stat-badge">
                <span className="stat-value">{makespan.toFixed(0)}h</span>
                <span className="stat-label">Makespan</span>
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
      <div className="machines-chart-container">
        <div className="chart-wrapper">
          <MachineGanttChart machines={scheduleData.machines} timeRange={timeRange} />
        </div>
      </div>
    </div>
  );
};

export default MachinesFullPage;