# PPO Production Scheduler Frontend

A React TypeScript frontend for the PPO Production Scheduler system, providing visual schedule management and real-time monitoring.

## Features

- **Real-time Health Monitoring**: Live system status updates
- **Interactive Job Configuration**: Add, edit, and manage production jobs
- **Gantt Chart Visualization**: Visual timeline of scheduled jobs on machines
- **Performance Metrics Dashboard**: Key performance indicators including makespan, utilization, and completion rates
- **Mock Data Support**: Automatic fallback when API is unavailable

## Prerequisites

- Node.js 18+ and npm
- Backend API running on port 8000

## Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at http://localhost:3000

## Architecture

### Components

- **Dashboard**: Main container component managing application state
- **HealthStatus**: Displays API health and connection status
- **ScheduleForm**: Interactive form for job configuration
- **GanttChart**: Visual schedule representation
- **MetricsDisplay**: Performance metrics visualization

### Services

- **api.ts**: Centralized API client with error handling
- Proxy configuration for backend communication

## Development

```bash
# Start development server
npm start

# Build for production
npm build

# Preview production build
npm preview
```

## API Integration

The frontend connects to the backend API at `http://localhost:8000` with the following endpoints:
- `/health` - System health check
- `/schedule` - Create production schedules

## Configuration

- Vite configuration in `vite.config.ts`
- TypeScript configuration in `tsconfig.json`
- Proxy settings for API communication

## Technologies

- React 18 with TypeScript
- Material-UI for components
- Recharts for data visualization
- Axios for API communication
- Vite for fast development and building