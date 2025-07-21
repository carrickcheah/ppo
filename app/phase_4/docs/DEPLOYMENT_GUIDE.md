# Phase 4 Production Deployment Guide

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.12+
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4+ cores recommended

### Software Dependencies
- **UV Package Manager**: Latest version
- **MariaDB**: Access to production database
- **Docker**: Optional for containerized deployment

### Network Requirements
- **API Port**: 8000 (configurable)
- **Database Access**: MariaDB connection
- **Firewall**: Allow incoming connections to API port

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/your-org/ppo-scheduler.git
cd ppo-scheduler/app
```

### 2. Setup Python Environment
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv sync
```

### 3. Configure Environment Variables
```bash
# Create .env file
cp .env.example .env

# Edit with your configuration
vim .env
```

Required environment variables:
```bash
# Database Configuration
DB_HOST=your-mariadb-host
DB_PORT=3306
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=production_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key

# Model Configuration
MODEL_PATH=models/full_production/final_model.zip
BATCH_SIZE=170

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ppo_scheduler.log
```

### 4. Prepare Production Data
```bash
# Fetch latest production snapshot
uv run python src/data_ingestion/ingest_data.py \
    --output data/real_production_snapshot.json

# Verify data integrity
uv run python scripts/verify_production_data.py
```

### 5. Download Trained Model
```bash
# Create model directory
mkdir -p models/full_production

# Download trained model (or copy from training server)
cp /path/to/trained/final_model.zip models/full_production/

# Verify model
uv run python scripts/verify_model.py
```

## Configuration Details

### API Server Configuration

Create `configs/api_config.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  log_level: "info"

security:
  api_key_header: "X-API-Key"
  rate_limit: 100  # requests per minute
  cors_origins: ["*"]  # Configure for production

scheduler:
  model_path: "models/full_production/final_model.zip"
  batch_size: 170
  timeout: 30  # seconds
  fallback_enabled: true

logging:
  format: "json"
  level: "INFO"
  file: "logs/api.log"
  max_size: "100MB"
  backup_count: 5
```

### Database Configuration

Create `configs/database.yaml`:
```yaml
database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  user: ${DB_USER}
  password: ${DB_PASSWORD}
  database: ${DB_NAME}
  
connection_pool:
  min_size: 5
  max_size: 20
  timeout: 30
  retry_attempts: 3
  
tables:
  machines: "tbl_machine"
  jobs: "tbl_workorder"
  schedule: "tbl_schedule_output"
```

## Starting the Service

### Development Mode
```bash
# Start with auto-reload
uv run python run_api_server.py --reload
```

### Production Mode

#### Option 1: Direct Execution
```bash
# Start server
uv run python run_api_server.py

# Or with gunicorn
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

#### Option 2: Systemd Service

Create `/etc/systemd/system/ppo-scheduler.service`:
```ini
[Unit]
Description=PPO Production Scheduler API
After=network.target

[Service]
Type=exec
User=ppo-user
Group=ppo-group
WorkingDirectory=/opt/ppo-scheduler/app
Environment="PATH=/opt/ppo-scheduler/app/.venv/bin"
EnvironmentFile=/opt/ppo-scheduler/app/.env
ExecStart=/opt/ppo-scheduler/app/.venv/bin/python run_api_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ppo-scheduler
sudo systemctl start ppo-scheduler
sudo systemctl status ppo-scheduler
```

#### Option 3: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY data/ data/

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run server
CMD ["uv", "run", "python", "run_api_server.py"]
```

Build and run:
```bash
# Build image
docker build -t ppo-scheduler:latest .

# Run container
docker run -d \
    --name ppo-scheduler \
    -p 8000:8000 \
    --env-file .env \
    ppo-scheduler:latest
```

## Testing Procedures

### 1. Health Check
```bash
curl http://localhost:8000/health

# Expected response:
{
    "status": "healthy",
    "model_loaded": true,
    "database_connected": true,
    "version": "1.0.0"
}
```

### 2. Test Scheduling Request
```bash
# Small test (should complete in single batch)
curl -X POST http://localhost:8000/schedule \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your-api-key" \
    -d @test_data/small_job_set.json

# Large test (triggers batch processing)
curl -X POST http://localhost:8000/schedule \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your-api-key" \
    -d @test_data/large_job_set.json
```

### 3. Integration Test Suite
```bash
# Run all integration tests
uv run pytest tests/integration/ -v

# Run specific test
uv run pytest tests/integration/test_api_endpoints.py::test_batch_scheduling -v
```

### 4. Load Testing
```bash
# Install locust
uv add --dev locust

# Run load test
uv run locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --users 10 \
    --spawn-rate 2
```

### 5. Smoke Tests
```bash
# Run smoke test script
uv run python scripts/smoke_test.py

# Checks:
# - API responds to health check
# - Can schedule 10 test jobs
# - Response time < 1 second
# - All jobs get scheduled
# - No errors in logs
```

## Monitoring Setup

### 1. Application Metrics

#### Prometheus Configuration

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'ppo_scheduler'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

#### Key Metrics to Monitor
- `scheduler_requests_total` - Total scheduling requests
- `scheduler_request_duration_seconds` - Request latency
- `scheduler_jobs_scheduled_total` - Jobs successfully scheduled
- `scheduler_batch_processing_total` - Batch processing events
- `scheduler_errors_total` - Error count by type

### 2. Log Aggregation

#### Filebeat Configuration
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /opt/ppo-scheduler/app/logs/*.log
  json.keys_under_root: true
  json.add_error_key: true
  fields:
    service: ppo-scheduler
    environment: production
```

### 3. Alerting Rules

Create `alerts.yml`:
```yaml
groups:
  - name: ppo_scheduler
    rules:
      - alert: HighErrorRate
        expr: rate(scheduler_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: SlowScheduling
        expr: scheduler_request_duration_seconds{quantile="0.95"} > 5
        for: 10m
        annotations:
          summary: "Scheduling requests taking too long"
          
      - alert: LowCompletionRate
        expr: scheduler_completion_rate < 0.95
        for: 15m
        annotations:
          summary: "Job completion rate below threshold"
```

### 4. Dashboard Setup

Grafana dashboard JSON available at `monitoring/dashboards/ppo_scheduler.json`

Key panels:
- Request rate and latency
- Job scheduling success rate
- Batch processing frequency
- Resource utilization
- Error rates by type

## Troubleshooting

### Common Issues

#### 1. Model Loading Failure
```
Error: Could not load model from path
```
**Solution**: 
- Verify model file exists and has correct permissions
- Check model path in configuration
- Ensure model version matches code version

#### 2. Database Connection Error
```
Error: Can't connect to MariaDB server
```
**Solution**:
- Verify database credentials in .env
- Check network connectivity to database
- Ensure database user has required permissions

#### 3. High Memory Usage
```
Warning: Memory usage exceeding threshold
```
**Solution**:
- Reduce batch size in configuration
- Increase server memory
- Enable memory profiling to identify leaks

#### 4. Slow Scheduling Performance
```
Warning: Scheduling taking longer than expected
```
**Solution**:
- Check CPU utilization
- Verify model is using appropriate device (CPU/GPU)
- Consider reducing batch size
- Enable performance profiling

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG uv run python run_api_server.py
```

This provides:
- Detailed request/response logging
- Environment state transitions
- Model inference details
- Batch processing steps

## Maintenance Procedures

### Daily Tasks
1. Check system health via monitoring dashboard
2. Review error logs for anomalies
3. Verify scheduling completion rates
4. Monitor resource utilization

### Weekly Tasks
1. Analyze performance trends
2. Review and rotate logs
3. Update production data snapshot
4. Check for model drift

### Monthly Tasks
1. Performance optimization review
2. Security audit (API keys, access logs)
3. Database maintenance (indexes, cleanup)
4. Model retraining evaluation

## Rollback Procedures

### Quick Rollback
```bash
# Stop current service
sudo systemctl stop ppo-scheduler

# Restore previous model
cp models/full_production/backup/final_model.zip models/full_production/

# Restart service
sudo systemctl start ppo-scheduler

# Verify functionality
curl http://localhost:8000/health
```

### Full Rollback
```bash
# Tag current version
git tag -a v1.0.1-rollback -m "Rollback point"

# Checkout previous stable version
git checkout v1.0.0

# Rebuild and deploy
uv sync
sudo systemctl restart ppo-scheduler
```

## Security Checklist

- [ ] API keys rotated and stored securely
- [ ] Database credentials encrypted
- [ ] HTTPS enabled for production
- [ ] Rate limiting configured
- [ ] Input validation enabled
- [ ] Audit logging active
- [ ] Firewall rules configured
- [ ] Regular security updates applied

## Support Information

### Documentation
- Technical Documentation: `/docs/technical/`
- API Reference: `/docs/api/`
- Troubleshooting Guide: `/docs/troubleshooting/`

### Contacts
- Development Team: dev-team@company.com
- Operations: ops@company.com
- On-call: +1-555-0123 (24/7)

### Useful Commands
```bash
# View logs
journalctl -u ppo-scheduler -f

# Check service status
systemctl status ppo-scheduler

# Test scheduling
uv run python scripts/test_scheduling.py

# Database connectivity
uv run python scripts/test_db_connection.py
```