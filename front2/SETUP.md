# Front2 PPO Integration Setup Guide

## Prerequisites

1. Node.js (v16 or higher)
2. PPO backend running on port 8000
3. MariaDB database with production data

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/carrickcheah/Project/ppo/front2
npm install
```

### 2. Configure Environment

Create a `.env` file in the front2 directory:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# PPO Backend URL
VITE_PPO_API_URL=http://localhost:8000
VITE_USE_PPO_BACKEND=true

# Optional: Supabase for authentication
VITE_SUPABASE_URL=your_supabase_url_here
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

### 3. Start PPO Backend

In a new terminal:

```bash
cd /Users/carrickcheah/Project/ppo/app
python scripts/run_api_server.py
```

Verify the backend is running:
- Open http://localhost:8000/docs
- Check http://localhost:8000/health

### 4. Start Frontend

```bash
cd /Users/carrickcheah/Project/ppo/front2
npm run dev
```

The application will be available at http://localhost:3000

## Usage

1. Open http://localhost:3000 in your browser
2. If authentication is enabled, log in with your credentials
3. Navigate to the Dashboard
4. Click "Refresh All Data" to load schedule from PPO backend
5. View the schedule in various formats:
   - Schedule Table
   - Jobs Allocation (Gantt Chart)
   - Machine Allocation (Resource View)
   - AI Report

## Troubleshooting

### "Cannot connect to API server"
- Ensure PPO backend is running on port 8000
- Check browser console for specific errors
- Verify CORS is properly configured

### "No pending jobs found"
- Check database connection in PPO backend
- Ensure there are unscheduled jobs in the database
- Check PPO backend logs for database queries

### Authentication Issues
- If not using Supabase, the app will show a warning but continue
- To disable authentication, you may need to modify AuthContext
- Ensure Supabase credentials are correct in .env

### Build for Production

```bash
npm run build
npm run preview
```

The production build will be in the `dist` directory.