import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AppBar, Tabs, Tab, Box, Container, Toolbar, Typography } from '@mui/material';
import Dashboard from './components/Dashboard';
import JobsChart from './components/JobsChart';
import MachineChart from './components/MachineChart';
import { ScheduleProvider } from './context/ScheduleContext';
import './styles/App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [value, setValue] = useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ScheduleProvider>
        <div className="App">
          <AppBar position="static" color="primary">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                PPO Production Scheduler
              </Typography>
            </Toolbar>
          </AppBar>
          
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={value} onChange={handleChange} aria-label="scheduler tabs">
              <Tab label="Dashboard" />
              <Tab label="Jobs Chart" />
              <Tab label="Machine Chart" />
            </Tabs>
          </Box>

          <Container maxWidth={false} sx={{ mt: 2 }}>
            <TabPanel value={value} index={0}>
              <Dashboard />
            </TabPanel>
            <TabPanel value={value} index={1}>
              <JobsChart />
            </TabPanel>
            <TabPanel value={value} index={2}>
              <MachineChart />
            </TabPanel>
          </Container>
        </div>
      </ScheduleProvider>
    </ThemeProvider>
  );
}

export default App;