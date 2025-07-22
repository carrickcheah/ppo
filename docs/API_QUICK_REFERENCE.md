# PPO Scheduler API - Quick Reference

## Start Server
```bash
cd /Users/carrickcheah/Project/ppo/app
uv run python run_api_server.py
```

## Key URLs
- API Base: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

## Authentication
Add to all requests:
```
X-API-Key: dev-api-key-change-in-production
```

## Schedule Jobs

**Endpoint:** POST /schedule

**Quick Example:**
```bash
curl -X POST http://localhost:8000/schedule \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d @jobs.json
```

**Sample jobs.json:**
```json
{
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
}
```

## Job Priority
- 1 = High (urgent)
- 2 = Medium (normal)
- 3 = Low (flexible)

## Machine Types
- 1 = Type 1 machines
- 2 = Type 2 machines
- 3 = Type 3 machines
- 4 = Type 4 machines

## Response Metrics
- **makespan**: Total schedule duration (hours)
- **completion_rate**: % of jobs scheduled
- **average_utilization**: Machine efficiency %
- **important_jobs_on_time**: % meeting LCD

## Common Issues

**500 Error**: Database/machine data issue
**422 Error**: Invalid job data format
**403 Error**: Invalid API key

## Performance Tips
- Batch size: Max 170 jobs per request
- Response time: 1-3 seconds typical
- For >170 jobs: Split into multiple requests

## Model Info
- Current: Phase 4 (49.2h makespan, 100% completion)
- Location: `models/full_production/final_model.zip`