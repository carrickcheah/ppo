# PPO Scheduler - Operator Quick Reference

## System Overview

The PPO Scheduler uses artificial intelligence to automatically schedule production jobs across machines. It processes jobs in batches of 170 to ensure 100% completion.

## Basic Operations

### Starting the System

```bash
# Navigate to application directory
cd /opt/ppo-scheduler/app

# Activate environment
source .venv/bin/activate

# Start API server
python run_api_server.py
```

### Stopping the System

```bash
# Graceful shutdown
Ctrl+C in the terminal

# Or if using systemd
sudo systemctl stop ppo-scheduler
```

### Checking System Status

```bash
# Health check
curl http://localhost:8000/health

# System logs
tail -f logs/api_server.log

# Service status (if using systemd)
sudo systemctl status ppo-scheduler
```

## Common Tasks

### 1. Schedule Production Jobs

**Via Command Line:**
```bash
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d @production_jobs.json
```

**Expected Response:**
```json
{
  "schedule_id": "sch_20250721_001",
  "scheduled_jobs": [...],
  "metrics": {
    "total_jobs": 411,
    "scheduled_jobs": 411,
    "completion_rate": 100.0,
    "makespan": 47.8,
    "batches_used": 3
  }
}
```

### 2. Update Production Data

```bash
# Fetch latest data from database
python src/data_ingestion/ingest_data.py \
  --output data/real_production_snapshot.json

# Restart server to use new data
sudo systemctl restart ppo-scheduler
```

### 3. Monitor Performance

**Check recent schedules:**
```bash
# View last 10 scheduling events
grep "Schedule created" logs/api_server.log | tail -10
```

**Check for errors:**
```bash
# View recent errors
grep "ERROR" logs/api_server.log | tail -20
```

## Understanding the Output

### Schedule Metrics

- **completion_rate**: Should always be 100%
- **makespan**: Total time to complete all jobs (hours)
- **batches_used**: Number of 170-job batches processed
- **scheduled_jobs**: Total jobs successfully scheduled

### Job Assignment

Each scheduled job contains:
- `job_id`: Unique identifier (e.g., "JOST25060144")
- `machine_id`: Assigned machine (e.g., "CM03")
- `start_time`: When job begins
- `end_time`: When job completes
- `processing_time`: Duration in hours

## Troubleshooting

### Issue: "No jobs scheduled"

**Check:**
1. Production data is loaded
2. Jobs have valid machine types
3. API request format is correct

**Fix:**
```bash
# Reload production data
python src/data_ingestion/ingest_data.py
# Restart server
sudo systemctl restart ppo-scheduler
```

### Issue: "API not responding"

**Check:**
1. Server is running
2. Port 8000 is not blocked
3. No other service using port

**Fix:**
```bash
# Check if running
ps aux | grep run_api_server
# Check port
netstat -an | grep 8000
# Restart if needed
sudo systemctl restart ppo-scheduler
```

### Issue: "Slow scheduling"

**Check:**
1. Number of jobs (>400 triggers batching)
2. CPU usage
3. Memory availability

**Fix:**
- Normal for 400+ jobs (batch processing)
- Ensure adequate CPU/memory
- Check logs for bottlenecks

### Issue: "Invalid job data"

**Requirements:**
- `processing_time` > 0
- `machine_types` not empty
- `lcd_date` in future
- Valid `job_id`

## Batch Processing

### When It Happens
- Automatically when >170 jobs submitted
- Transparent to users
- Jobs prioritized by importance and LCD

### What You'll See
```json
{
  "metrics": {
    "batches_used": 3,
    "warning": "Large job set processed in 3 batches"
  }
}
```

### Why It's Normal
- Environment limitation requires batching
- Ensures 100% completion
- Minimal impact on quality

## Best Practices

### DO:
- Update production data daily
- Monitor completion rates
- Check logs for warnings
- Keep API key secure
- Document any issues

### DON'T:
- Modify the model files
- Change batch size without testing
- Ignore error messages
- Skip data updates
- Share API keys

## Quick Commands

```bash
# Start server
python run_api_server.py

# Check health
curl http://localhost:8000/health

# View logs
tail -f logs/api_server.log

# Update data
python src/data_ingestion/ingest_data.py

# Test scheduling
python test_api_integration.py

# Check metrics
grep "metrics" logs/api_server.log | tail -5
```

## Emergency Contacts

- **System Administrator**: ext. 1234
- **Development Team**: dev-support@company.com
- **24/7 Support**: +1-555-0123

## Fallback Procedure

If PPO scheduler fails:
1. Note the error message
2. Switch to manual scheduling
3. Contact support team
4. Preserve log files

---

**Version**: 1.0.0  
**Last Updated**: July 2025  
**Documentation**: `/docs/full_documentation.md`