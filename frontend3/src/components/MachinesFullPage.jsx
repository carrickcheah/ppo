import React from 'react';
import MachineGanttChart from './MachineGanttChart';

const MachinesFullPage = ({ scheduleData, timeRange, setTimeRange }) => {
  if (!scheduleData) {
    return (
      <div className="fullpage-container">
        <div className="no-data-message">
          <h2>No Schedule Data Available</h2>
          <p>Please go to the Dashboard and schedule jobs first.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fullpage-container">
      <div className="fullpage-header">
        <h2>Machine Allocation Chart</h2>
        <div className="fullpage-controls">
          <div className="control-group">
            <label htmlFor="time-range-fullpage">Time Range:</label>
            <select 
              id="time-range-fullpage"
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
      </div>
      <div className="fullpage-chart">
        <MachineGanttChart machines={scheduleData.machines} timeRange={timeRange} />
      </div>
    </div>
  );
};

export default MachinesFullPage;