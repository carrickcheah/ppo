# PPO Backend Integration for Front2

This document describes the integration between the front2 React frontend and the PPO (Proximal Policy Optimization) backend scheduler.

## Overview

The front2 application has been modified to connect with the PPO-based scheduling backend instead of the constraint programming solver. This integration maintains the existing UI while leveraging the machine learning-based optimization from the PPO model.

## Architecture Changes

### 1. New Services Created

- **ppoApi.ts**: Handles communication with the PPO backend API
  - Health checks
  - Schedule creation
  - Job loading from database
  
- **ppoAdapter.ts**: Transforms PPO responses to match existing component formats
  - Converts to Gantt priority view
  - Converts to Gantt resource view
  - Creates detailed schedule format
  - Generates schedule overview metrics
  
- **jobDataService.ts**: Enriches scheduled jobs with metadata
  - Fetches actual job data including LCD dates
  - Provides accurate buffer status calculation

### 2. Configuration Updates

- **config.ts**: Added PPO backend configuration
  - `PPO_API_URL`: PPO backend URL (default: http://localhost:8000)
  - `USE_PPO_BACKEND`: Toggle between PPO and constraint programming (default: true)

- **vite.config.js**: Added proxy for PPO backend
  - `/ppo` proxy routes to PPO backend

### 3. DataCacheContext Updates

The DataCacheContext now:
1. Calls PPO backend `/schedule` endpoint
2. Transforms the response using ppoAdapter
3. Maintains the same interface for components

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the front2 directory:

```env
# PPO Backend Configuration
VITE_PPO_API_URL=http://localhost:8000
VITE_USE_PPO_BACKEND=true

# Supabase (for authentication)
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 2. Start the PPO Backend

```bash
cd /Users/carrickcheah/Project/ppo/app
python scripts/run_api_server.py
```

The PPO backend should be running on http://localhost:8000

### 3. Start the Frontend

```bash
cd /Users/carrickcheah/Project/ppo/front2
npm install
npm run dev
```

The frontend will be available at http://localhost:3000

## How It Works

### Data Flow

1. **User clicks "Refresh All Data"** in Dashboard
2. **DataCacheContext** calls PPO backend:
   - `GET /health` - Check backend status
   - `POST /schedule` - Create schedule (loads jobs from database)
   - Optionally fetches job metadata
3. **ppoAdapter** transforms the response:
   - Scheduled jobs → Gantt views
   - Metrics → Schedule overview
   - Calculates buffer status based on LCD dates
4. **Components** display the transformed data:
   - GanttChartDisplay shows timeline
   - DetailedScheduleTable shows job details
   - Dashboard shows metrics

### Key Transformations

#### Buffer Status Calculation
- **Late**: LCD date has passed
- **Warning**: Less than 24 hours until LCD
- **Caution**: Less than 72 hours until LCD  
- **OK**: More than 72 hours until LCD

#### Priority Mapping
- Uses original job priority if available
- Defaults to priority 2 for unknown jobs

## API Endpoints

### PPO Backend Endpoints Used

- `GET /health` - System health check
- `POST /schedule` - Create optimized schedule
  - Can provide jobs array or load from database
  - Returns scheduled jobs with timings

### Expected Response Format

```typescript
{
  schedule_id: string,
  scheduled_jobs: [{
    job_id: string,
    machine_id: number,
    machine_name: string,
    start_time: number,
    end_time: number,
    start_datetime: string,
    end_datetime: string,
    setup_time_included: number
  }],
  metrics: {
    makespan: number,
    total_jobs: number,
    scheduled_jobs: number,
    completion_rate: number,
    average_utilization: number,
    total_setup_time: number,
    important_jobs_on_time: number
  },
  timestamp: string
}
```

## Troubleshooting

### PPO Backend Not Connected
- Check if PPO backend is running on port 8000
- Verify database connection in PPO backend
- Check browser console for CORS errors

### No Data Displayed
- Ensure PPO backend has access to production database
- Check if there are pending jobs in the database
- Verify DataCacheContext refresh completes without errors

### Buffer Status Incorrect
- Verify jobDataService is fetching correct LCD dates
- Check if job enrichment is working properly
- Ensure date calculations account for timezone

## Future Enhancements

1. Real-time schedule updates
2. Manual job assignment override
3. Schedule comparison between PPO and constraint programming
4. Historical performance tracking
5. Integration with production monitoring systems