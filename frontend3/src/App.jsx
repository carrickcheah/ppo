import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import './App.css';
import schedulingApi from './services/api';
import Dashboard from './components/Dashboard';
import JobsFullPage from './components/JobsFullPage';
import MachinesFullPage from './components/MachinesFullPage';

function Navigation() {
  const location = useLocation();
  
  return (
    <nav className="main-navigation">
      <Link 
        to="/" 
        className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
      >
        Dashboard
      </Link>
      <Link 
        to="/jobs" 
        className={`nav-link ${location.pathname === '/jobs' ? 'active' : ''}`}
      >
        Jobs View
      </Link>
      <Link 
        to="/machines" 
        className={`nav-link ${location.pathname === '/machines' ? 'active' : ''}`}
      >
        Machines View
      </Link>
    </nav>
  );
}

function App() {
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('100_jobs');
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scheduleData, setScheduleData] = useState(null);
  const [activeChart, setActiveChart] = useState('jobs');
  const [timeRange, setTimeRange] = useState('4weeks');
  const [isFullscreen, setIsFullscreen] = useState(false);

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
      
      // Set default model if available
      if (modelsRes.models && modelsRes.models.length > 0) {
        // Prefer sb3_1million if available, otherwise use first model
        const defaultModel = modelsRes.models.find(m => m.name === 'sb3_1million') || modelsRes.models[0];
        setSelectedModel(defaultModel.name);
      }
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
    <Router>
      <div className="App">
        <Navigation />

        <Routes>
          <Route 
            path="/" 
            element={
              <Dashboard
                scheduleData={scheduleData}
                loading={loading}
                error={error}
                selectedDataset={selectedDataset}
                setSelectedDataset={setSelectedDataset}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                timeRange={timeRange}
                setTimeRange={setTimeRange}
                handleSchedule={handleSchedule}
                activeChart={activeChart}
                setActiveChart={setActiveChart}
                isFullscreen={isFullscreen}
                setIsFullscreen={setIsFullscreen}
                models={models}
              />
            } 
          />
          <Route 
            path="/jobs" 
            element={
              <JobsFullPage
                scheduleData={scheduleData}
                timeRange={timeRange}
                setTimeRange={setTimeRange}
              />
            } 
          />
          <Route 
            path="/machines" 
            element={
              <MachinesFullPage
                scheduleData={scheduleData}
                timeRange={timeRange}
                setTimeRange={setTimeRange}
              />
            } 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;