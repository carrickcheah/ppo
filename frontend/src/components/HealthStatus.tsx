import React from 'react';
import {
  Box,
  Typography,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Divider,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import RefreshIcon from '@mui/icons-material/Refresh';
import StorageIcon from '@mui/icons-material/Storage';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import { HealthResponse } from '../types';

interface HealthStatusProps {
  health: HealthResponse | null;
  onRefresh: () => void;
}

const HealthStatus: React.FC<HealthStatusProps> = ({ health, onRefresh }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon color="success" />;
      case 'degraded':
        return <WarningIcon color="warning" />;
      case 'unhealthy':
        return <ErrorIcon color="error" />;
      default:
        return <ErrorIcon color="disabled" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (!health) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          System Status
        </Typography>
        <Typography color="text.secondary">
          Connecting to API...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" gutterBottom>
          System Status
        </Typography>
        <IconButton size="small" onClick={onRefresh} title="Refresh">
          <RefreshIcon />
        </IconButton>
      </Box>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        {getStatusIcon(health.status)}
        <Chip
          label={health.status.toUpperCase()}
          color={getStatusColor(health.status) as any}
          size="small"
          sx={{ ml: 1 }}
        />
      </Box>

      <Divider sx={{ my: 1 }} />

      <List dense>
        <ListItem>
          <ListItemIcon>
            <ModelTrainingIcon />
          </ListItemIcon>
          <ListItemText
            primary="Model"
            secondary={health.model_loaded ? 'Loaded' : 'Not loaded'}
          />
          {health.model_loaded ? (
            <CheckCircleIcon color="success" fontSize="small" />
          ) : (
            <ErrorIcon color="error" fontSize="small" />
          )}
        </ListItem>

        <ListItem>
          <ListItemIcon>
            <StorageIcon />
          </ListItemIcon>
          <ListItemText
            primary="Database"
            secondary={health.database_connected ? 'Connected' : 'Disconnected'}
          />
          {health.database_connected ? (
            <CheckCircleIcon color="success" fontSize="small" />
          ) : (
            <WarningIcon color="warning" fontSize="small" />
          )}
        </ListItem>

        <ListItem>
          <ListItemIcon>
            <AccessTimeIcon />
          </ListItemIcon>
          <ListItemText
            primary="Uptime"
            secondary={formatUptime(health.uptime)}
          />
        </ListItem>
      </List>

      <Divider sx={{ my: 1 }} />

      <Typography variant="caption" color="text.secondary">
        Version: {health.version}
      </Typography>
    </Box>
  );
};

export default HealthStatus;