# PPO Scheduler API Usage Guide

## Overview

The PPO Scheduler API provides a REST interface for scheduling production jobs using our trained Phase 4 model. The API is designed to integrate seamlessly with existing production systems and provides real-time scheduling decisions.

## Quick Start

### 1. Start the API Server

```bash
cd /Users/carrickcheah/Project/ppo/app
uv run python run_api_server.py
```

The server will start on `http://localhost:8000` by default.

### 2. Check API Health

```bash
curl -X GET http://localhost:8000/health \
  -H "X-API-Key: dev-api-key-change-in-production"
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "database_connected": true,
  "uptime": 123.45,
  "environment": "development"
}
```

### 3. Access API Documentation

Open your browser and navigate to: `http://localhost:8000/docs`

This provides an interactive Swagger UI where you can test all endpoints.

## API Endpoints

### Health Check - GET /health

Check if the API is running and the model is loaded.

**Headers:**
- `X-API-Key`: Your API key

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "version": "1.0.0",
  "model_loaded": true|false,
  "database_connected": true|false,
  "uptime": 123.45,
  "last_schedule_time": "2025-07-22T14:30:00",
  "environment": "development|staging|production"
}
```

### Schedule Jobs - POST /schedule

Generate an optimized schedule for a batch of jobs.

**Headers:**
- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body:**
```json
{
  "jobs": [
    {
      "job_id": "JOAW001",
      "family_id": "FAM001",
      "sequence": 1,
      "processing_time": 2.5,
      "machine_types": [1, 2, 3],
      "priority": 2,
      "is_important": true,
      "lcd_date": "2025-07-25T00:00:00",
      "setup_time": 0.3
    }
  ],
  "schedule_start": "2025-07-22T14:00:00"
}
```

**Response:**
```json
{
  "schedule_id": "uuid-here",
  "scheduled_jobs": [
    {
      "job_id": "JOAW001",
      "machine_id": 42,
      "machine_name": "CM03",
      "start_time": 0.0,
      "end_time": 2.8,
      "start_datetime": "2025-07-22T14:00:00",
      "end_datetime": "2025-07-22T16:48:00",
      "setup_time_included": 0.3
    }
  ],
  "metrics": {
    "makespan": 49.2,
    "total_jobs": 100,
    "scheduled_jobs": 100,
    "completion_rate": 100.0,
    "average_utilization": 78.5,
    "total_setup_time": 30.0,
    "important_jobs_on_time": 95.0
  },
  "timestamp": "2025-07-22T14:30:00"
}
```

## Job Parameters

Each job in the request must include:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| job_id | string | Unique job identifier | "JOAW001" |
| family_id | string | Job family for grouping | "FAM001" |
| sequence | integer | Sequence within family | 1 |
| processing_time | float | Hours to process | 2.5 |
| machine_types | array[int] | Compatible machine types | [1, 2, 3] |
| priority | integer | Priority (1=high, 3=low) | 2 |
| is_important | boolean | Important job flag | true |
| lcd_date | string (ISO) | Latest completion date | "2025-07-25T00:00:00" |
| setup_time | float | Setup time in hours | 0.3 |

## Authentication

All API requests require an API key in the headers:

```
X-API-Key: your-api-key-here
```

For development: `dev-api-key-change-in-production`

## Environment Variables

Configure the API via environment variables in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key
ENVIRONMENT=production

# Database Configuration
DB_HOST=your-mariadb-host
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=nex_valiant
DB_PORT=3306

# Model Configuration (optional)
MODEL_PATH=models/full_production/final_model.zip
```

## Integration Examples

### Python Example

```python
import requests
import json
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

# Prepare job data
jobs = [
    {
        "job_id": "JOAW001",
        "family_id": "FAM001",
        "sequence": 1,
        "processing_time": 2.5,
        "machine_types": [1, 2, 3],
        "priority": 2,
        "is_important": True,
        "lcd_date": (datetime.now() + timedelta(days=3)).isoformat(),
        "setup_time": 0.3
    }
]

# Make request
response = requests.post(
    f"{API_URL}/schedule",
    headers={
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    },
    json={
        "jobs": jobs,
        "schedule_start": datetime.now().isoformat()
    }
)

if response.status_code == 200:
    schedule = response.json()
    print(f"Makespan: {schedule['metrics']['makespan']} hours")
    print(f"Scheduled: {schedule['metrics']['scheduled_jobs']} jobs")
else:
    print(f"Error: {response.status_code}")
```

### Curl Example

```bash
curl -X POST http://localhost:8000/schedule \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jobs": [{
      "job_id": "JOAW001",
      "family_id": "FAM001",
      "sequence": 1,
      "processing_time": 2.5,
      "machine_types": [1, 2, 3],
      "priority": 2,
      "is_important": true,
      "lcd_date": "2025-07-25T00:00:00",
      "setup_time": 0.3
    }],
    "schedule_start": "2025-07-22T14:00:00"
  }'
```

## Performance Guidelines

1. **Batch Size**: The API can handle up to 170 jobs per request due to environment limitations
2. **Response Time**: Expect 1-3 seconds for typical batches (50-100 jobs)
3. **Concurrent Requests**: The API can handle multiple concurrent requests
4. **Rate Limiting**: No built-in rate limiting in development mode

## Model Updates

To update the scheduling model:

1. Train a new model and save it as a `.zip` file
2. Replace the model file at `models/full_production/final_model.zip`
3. Restart the API server
4. The new model will be loaded automatically

Alternative: Set `MODEL_PATH` environment variable to use a different model location.

## Troubleshooting

### API Returns 500 Error

Common causes:
1. Database connection failed - check DB credentials in `.env`
2. No machines in database - ensure machine data is loaded
3. Model file not found - verify model path exists

### API Returns 422 Error

This indicates invalid request data. Check:
1. All required job fields are present
2. Data types are correct (e.g., `processing_time` is a number)
3. ISO date format for `lcd_date` and `schedule_start`

### Model Not Loading

1. Check the model file exists: `ls models/full_production/final_model.zip`
2. Verify file permissions
3. Check server logs for detailed error messages

## Production Deployment

For production deployment:

1. **Security**: Change the default API key
2. **Database**: Use production MariaDB credentials
3. **Monitoring**: Enable metrics endpoint (port 9090)
4. **Scaling**: Use multiple workers: `--workers 4`
5. **HTTPS**: Deploy behind a reverse proxy (nginx/Apache)

## Support

For issues or questions:
- Check API logs: `logs/api_server.log`
- View API documentation: `http://localhost:8000/docs`
- Review server console output for real-time debugging

## Phase 4 Model Performance

The deployed Phase 4 model achieves:
- **Makespan**: 49.2 hours (average)
- **Completion Rate**: 100%
- **Utilization**: 78-82%
- **LCD Compliance**: >95% for important jobs

This represents a significant improvement over manual scheduling methods.