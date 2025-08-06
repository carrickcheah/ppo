# PPO Scheduler API Documentation

## Overview

The PPO Scheduler API provides REST endpoints for scheduling production jobs using trained Proximal Policy Optimization (PPO) models. The API is built with FastAPI and supports both direct JSON requests and file uploads.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. In production, implement API key or OAuth2 authentication.

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API service is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-06T10:30:00",
  "model_loaded": true
}
```

### 2. Root Information

**GET** `/`

Get basic API information and available endpoints.

**Response:**
```json
{
  "service": "PPO Scheduler API",
  "status": "running",
  "model_loaded": true
}
```

### 3. Model Information

**GET** `/model/info`

Get information about the loaded PPO model.

**Response:**
```json
{
  "model_loaded": true,
  "model_path": "checkpoints/fast/model_40jobs.pth",
  "capacity": "Up to 40 job families (127 tasks)",
  "performance": {
    "completion_rate": "92.9%",
    "average_reward": 27150,
    "training_time": "54 seconds",
    "inference_time": "<1 second"
  }
}
```

### 4. Schedule Jobs

**POST** `/schedule`

Schedule production jobs using the trained PPO model.

**Request Body:**
```json
{
  "families": {
    "JOST25060084": {
      "lcd_date": "2025-08-15",
      "lcd_days_remaining": 10.5,
      "tasks": [
        {
          "sequence": 1,
          "process_name": "CP08-056-1/2",
          "processing_time": 4.5,
          "assigned_machine": "PP09-160T-C-A1"
        },
        {
          "sequence": 2,
          "process_name": "CP08-056-2/2",
          "processing_time": 3.2,
          "assigned_machine": "PP09-160T-C-A1"
        }
      ]
    }
  },
  "machines": ["PP09-160T-C-A1", "WH01A-PK", "CL02"],
  "horizon_days": 30
}
```

**Response:**
```json
{
  "schedule": [
    {
      "task_id": 0,
      "family_id": "JOST25060084",
      "sequence": 1,
      "machine": "PP09-160T-C-A1",
      "start_time": 0,
      "end_time": 4.5,
      "processing_time": 4.5
    },
    {
      "task_id": 1,
      "family_id": "JOST25060084",
      "sequence": 2,
      "machine": "PP09-160T-C-A1",
      "start_time": 4.5,
      "end_time": 7.7,
      "processing_time": 3.2
    }
  ],
  "metrics": {
    "total_tasks": 2,
    "tasks_scheduled": 2,
    "completion_rate": "100.0%",
    "makespan": 7.7,
    "steps_taken": 2
  },
  "success": true,
  "message": "Successfully scheduled 2 out of 2 tasks"
}
```

### 5. Schedule from File

**POST** `/schedule/file`

Upload a JSON file containing job data for scheduling.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: File upload (JSON format)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/schedule/file" \
  -F "file=@jobs.json"
```

**Response:** Same as `/schedule` endpoint

### 6. Reload Model

**POST** `/model/reload`

Reload the PPO model from disk (useful after training new model).

**Response:**
```json
{
  "success": true,
  "message": "Model reloaded from checkpoints/fast/model_40jobs.pth"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Too many tasks (150). Model supports up to 127 tasks."
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded. Please check server logs."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Scheduling failed: [error message]"
}
```

## Data Format

### Task Structure
```json
{
  "sequence": 1,
  "process_name": "Process name with sequence info",
  "processing_time": 4.5,
  "assigned_machine": "Machine ID (optional)"
}
```

### Family Structure
```json
{
  "lcd_date": "2025-08-15",
  "lcd_days_remaining": 10.5,
  "tasks": [/* array of tasks */]
}
```

## Constraints

The scheduler respects the following constraints:

1. **Sequence Constraints**: Tasks within a family must complete in order (1/3 → 2/3 → 3/3)
2. **Machine Assignment**: Tasks use pre-assigned machines or any available machine
3. **No Overlap**: One task per machine at a time

## Performance

- **Inference Time**: < 1 second for 40 job families (127 tasks)
- **Capacity**: Up to 127 tasks per request
- **Completion Rate**: ~93% task scheduling success rate

## Running the API

### Local Development
```bash
cd /Users/carrickcheah/Project/ppo/app3
uv run python src/api/scheduler_api.py
```

### Docker
```bash
docker-compose up scheduler-api
```

### Production (with Nginx)
```bash
docker-compose --profile production up
```

## Examples

### Python Client
```python
import requests

url = "http://localhost:8000/schedule"
data = {
    "families": {
        "JOB001": {
            "lcd_date": "2025-08-20",
            "lcd_days_remaining": 14,
            "tasks": [
                {
                    "sequence": 1,
                    "process_name": "Cutting",
                    "processing_time": 2.5,
                    "assigned_machine": "CUT01"
                }
            ]
        }
    },
    "machines": ["CUT01", "CUT02"]
}

response = requests.post(url, json=data)
schedule = response.json()
print(f"Scheduled {schedule['metrics']['tasks_scheduled']} tasks")
```

### JavaScript Client
```javascript
const response = await fetch('http://localhost:8000/schedule', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data)
});

const schedule = await response.json();
console.log(`Scheduled ${schedule.metrics.tasks_scheduled} tasks`);
```

## Monitoring

- Health checks available at `/health`
- Model information at `/model/info`
- Logs available via Docker: `docker logs ppo-scheduler`

## Limitations

- Maximum 127 tasks per scheduling request
- Model trained on specific job patterns (JOST, JOTP, JOPRD prefixes)
- No real-time updates during scheduling
- Single model loaded at a time

## Future Enhancements

- Support for multiple models (different capacities)
- WebSocket support for real-time scheduling updates
- Batch scheduling for multiple requests
- Model versioning and A/B testing
- Authentication and rate limiting
- Caching for repeated requests