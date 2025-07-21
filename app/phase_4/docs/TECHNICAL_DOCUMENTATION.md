# Phase 4 Technical Documentation

## Architecture Overview

### System Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MariaDB       │────▶│  Data Ingestion  │────▶│ Production      │
│   Database      │     │  (ingest_data.py)│     │ Snapshot JSON   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│  PPO Scheduler   │────▶│ Full Production │
│   Server        │     │  (Batch Handler) │     │ Environment     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                           │
                                ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  PPO Model       │────▶│ Schedule Output │
                        │  (SB3)           │     │ (JSON)          │
                        └──────────────────┘     └─────────────────┘
```

### Data Flow

1. **Data Ingestion**: Real production data fetched from MariaDB
2. **Environment Setup**: Production snapshot loaded into FullProductionEnv
3. **API Request**: Jobs submitted via REST API
4. **Batch Processing**: Large job sets split into 170-job batches
5. **PPO Inference**: Model generates optimal schedule per batch
6. **Result Aggregation**: Batch schedules combined into final output

## Batch Scheduling Implementation

### The Challenge

The FullProductionEnv has a fundamental limitation:
- **Action Space**: job_idx × machine_idx (e.g., 411 × 30 = 12,330 actions)
- **Environment Limit**: max_valid_actions = 200-888
- **Result**: Only ~170 jobs visible per episode

### The Solution

```python
class PPOScheduler:
    def __init__(self, model_path: str, batch_size: int = 170):
        self.batch_size = batch_size
        self.model = PPO.load(model_path)
    
    def _schedule_in_batches(self, jobs: List[Job], machines: List[Machine]):
        # Sort jobs by priority and LCD date
        sorted_jobs = sorted(jobs, key=lambda j: (
            not j.is_important,  # Important jobs first
            j.lcd_date,          # Earlier deadlines first
            j.priority           # Higher priority first
        ))
        
        # Create batches
        batches = []
        for i in range(0, len(sorted_jobs), self.batch_size):
            batch = sorted_jobs[i:i + self.batch_size]
            batches.append(batch)
        
        # Schedule each batch
        all_scheduled_jobs = []
        for batch_idx, batch in enumerate(batches):
            scheduled = self._schedule_batch(batch, machines)
            all_scheduled_jobs.extend(scheduled)
        
        return all_scheduled_jobs
```

### Batch Processing Algorithm

1. **Job Prioritization**
   - Important jobs (is_important=True) scheduled first
   - Within importance level, sort by LCD date
   - Break ties using priority field

2. **Batch Creation**
   - Fixed batch size of 170 jobs
   - Last batch may be smaller
   - Maintains priority ordering

3. **Sequential Processing**
   - Each batch scheduled independently
   - Machine availability updated between batches
   - Start times adjusted for continuity

## Environment Design

### State Space (60 features)

```python
# Hierarchical state compression
state = [
    # Global features (10)
    total_remaining_time,
    avg_machine_utilization,
    important_jobs_ratio,
    avg_days_until_lcd,
    schedule_density,
    
    # Job features (25) - Top 5 jobs
    for job in top_5_jobs:
        job_processing_time,
        job_priority,
        job_is_important,
        days_until_lcd,
        compatible_machines_ratio,
    
    # Machine features (25) - Top 5 machines
    for machine in top_5_machines:
        machine_utilization,
        machine_load,
        time_until_available,
        compatible_jobs_ratio,
        efficiency_score
]
```

### Action Space

```python
# Discrete action space
action = job_idx * n_machines + machine_idx

# Action masking for validity
valid_actions = []
for job_idx in range(n_jobs):
    for machine_idx in compatible_machines[job_idx]:
        if machine_available(machine_idx):
            action = job_idx * n_machines + machine_idx
            valid_actions.append(action)
```

### Reward Function

```python
def calculate_reward(self):
    reward = 0
    
    # Completion reward
    if job_completed:
        reward += 10
        
        # Important job bonus
        if job.is_important:
            reward += 20
        
        # Efficiency bonus (early completion)
        if completion_time < deadline:
            reward += 5
    
    # Invalid action penalty
    if action_invalid:
        reward -= 20
    
    # Time penalty
    reward -= 0.1  # Per timestep
    
    # Episode completion bonus
    if all_jobs_scheduled:
        reward += 50
    
    return reward
```

## API Integration

### FastAPI Server Structure

```python
# Main endpoints
@app.post("/schedule")
async def create_schedule(request: ScheduleRequest) -> ScheduleResponse:
    # Validate request
    # Convert to internal format
    # Call PPO scheduler
    # Return formatted response

@app.get("/health")
async def health_check() -> HealthResponse:
    # Check model loaded
    # Check database connection
    # Return system status
```

### Request/Response Models

```python
class Job(BaseModel):
    job_id: str
    family_id: str
    processing_time: float
    machine_types: List[int]
    priority: int
    is_important: bool
    lcd_date: datetime

class ScheduleRequest(BaseModel):
    request_id: str
    jobs: List[Job]
    schedule_start: datetime

class ScheduledJob(BaseModel):
    job_id: str
    machine_id: str
    start_time: datetime
    end_time: datetime
    processing_time: float

class ScheduleResponse(BaseModel):
    schedule_id: str
    scheduled_jobs: List[ScheduledJob]
    metrics: ScheduleMetrics
    warnings: Optional[List[str]]
```

## Performance Analysis

### Scaling Characteristics

| Phase | Machines | Jobs | Makespan | Scaling Factor |
|-------|----------|------|----------|----------------|
| 2     | 10       | 172  | 86.3h    | Baseline       |
| 3     | 40       | 172  | 19.7h    | 4.4x improvement |
| 4     | 149      | 172  | 15.9h    | 1.2x improvement |

**Observation**: Sub-linear scaling with diminishing returns
- 4x machines → 4.4x speedup (super-linear due to better load distribution)
- 3.7x more machines → 1.2x speedup (diminishing returns)

### Batch Processing Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Jobs per batch | 170 | Environment limitation |
| Batches for 411 jobs | 3 | Automatic handling |
| Processing time per batch | <1s | Model inference |
| Total scheduling time | <3s | Including overhead |
| Memory usage | ~2GB | Stable across batches |

### Model Training Performance

```python
# Training configuration
config = {
    'total_timesteps': 1_000_000,
    'n_envs': 8,
    'batch_size': 512,
    'n_epochs': 10,
    'learning_rate': 1e-5,
    'training_time': '~10 minutes'
}

# Results
results = {
    'final_reward': 850,
    'completion_rate': 100%,
    'invalid_actions': <5%,
    'convergence': 'stable after 600k steps'
}
```

## Error Handling

### Environment Errors

1. **Invalid Action Handling**
   ```python
   if action >= self.action_space.n or action not in valid_actions:
       reward = -20
       self.invalid_action_count += 1
       return self.state, reward, False, False, info
   ```

2. **Data Loading Errors**
   ```python
   try:
       snapshot = load_production_snapshot()
   except FileNotFoundError:
       logger.error("Production snapshot not found")
       raise ValueError("Must run ingest_data.py first")
   ```

### API Error Responses

```python
# Validation error
{
    "error": "Invalid request",
    "detail": "Job processing_time must be positive",
    "status_code": 400
}

# Server error with fallback
{
    "error": "Scheduling failed",
    "detail": "PPO scheduler error, falling back to greedy",
    "fallback_used": true,
    "status_code": 500
}
```

## Configuration Management

### Environment Configuration

```yaml
# phase4_config.yaml
environment:
  n_machines: 150
  n_jobs: 500
  max_valid_actions: 888
  max_episode_steps: 3000
  state_compression: "hierarchical"
  use_break_constraints: true
  use_holiday_constraints: true

training:
  total_timesteps: 1000000
  n_envs: 8
  batch_size: 512
  learning_rate: 0.00001
  gamma: 0.99
  
api:
  batch_size: 170
  timeout: 30
  max_retries: 3
```

### Model Paths

```python
# Organized model storage
models/
├── toy/                 # Phase 1
├── medium/             # Phase 2
├── production/         # Phase 3
└── full_production/    # Phase 4
    ├── checkpoint_200k.zip
    ├── checkpoint_500k.zip
    ├── checkpoint_800k.zip
    └── final_model.zip
```

## Testing Strategy

### Unit Tests
```python
# Test batch scheduling logic
def test_batch_creation():
    jobs = create_test_jobs(500)
    scheduler = PPOScheduler(batch_size=170)
    batches = scheduler._create_batches(jobs)
    
    assert len(batches) == 3
    assert len(batches[0]) == 170
    assert len(batches[1]) == 170
    assert len(batches[2]) == 160
```

### Integration Tests
```python
# Test end-to-end scheduling
def test_full_scheduling():
    # Load real data
    jobs = load_production_jobs()
    
    # Schedule via API
    response = client.post("/schedule", json={"jobs": jobs})
    
    # Validate results
    assert response.status_code == 200
    assert response.json()["metrics"]["completion_rate"] == 100
```

### Performance Tests
```python
# Test scaling behavior
def test_scaling_performance():
    for n_jobs in [100, 200, 400, 800]:
        start_time = time.time()
        schedule = scheduler.schedule(create_jobs(n_jobs))
        elapsed = time.time() - start_time
        
        assert elapsed < n_jobs * 0.01  # <10ms per job
```

## Monitoring and Observability

### Key Metrics

1. **Scheduling Metrics**
   - Jobs scheduled per minute
   - Average makespan
   - Completion rate
   - Batch processing frequency

2. **Model Metrics**
   - Invalid action rate
   - Average episode reward
   - Inference time
   - Memory usage

3. **System Metrics**
   - API response time
   - Error rate
   - Database query time
   - CPU/Memory utilization

### Logging Strategy

```python
# Structured logging
logger.info("Schedule created", extra={
    "schedule_id": schedule_id,
    "total_jobs": len(jobs),
    "batches_used": n_batches,
    "makespan": makespan,
    "completion_rate": completion_rate,
    "processing_time_ms": elapsed_ms
})
```

## Security Considerations

### API Security
1. **Authentication**: API key required for all endpoints
2. **Rate Limiting**: 100 requests per minute per key
3. **Input Validation**: Strict schema validation
4. **SQL Injection**: Parameterized queries only

### Data Security
1. **No hardcoded credentials**: Environment variables only
2. **Encrypted connections**: TLS for API and database
3. **Audit logging**: All schedule requests logged
4. **Access control**: Role-based permissions

## Future Technical Improvements

### Short Term
1. **Caching**: Cache environment setup for faster inference
2. **Parallel Batch Processing**: Process batches concurrently
3. **GPU Acceleration**: Use GPU for larger models
4. **Prometheus Metrics**: Add detailed monitoring

### Long Term
1. **Hierarchical Action Space**: Solve 10,000+ action limitation
2. **Online Learning**: Continuous improvement from feedback
3. **Multi-Objective Optimization**: Balance multiple KPIs
4. **Distributed Training**: Scale to multiple factories