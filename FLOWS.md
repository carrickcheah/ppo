# PPO Scheduling System - Workflow Documentation

## System Overview

The PPO scheduling system uses deep reinforcement learning to optimize production scheduling. This document outlines the complete workflow from data ingestion to schedule visualization.

## app3 - Simplified PPO Scheduling System

### Overview
- **Simplified action space**: Select which task to schedule next (not job-machine pairs)
- **Pre-assigned machines**: 94% of tasks have specific machine assignments from database
- **Clean constraints**: Sequence, availability, material arrival only
- **Real production data**: JSON snapshots with 10-500 job families

### Data Structure
```json
{
  "families": {
    "JOST25060084": {
      "tasks": [
        {
          "sequence": 1,
          "process_name": "CP08-056-1/2",
          "processing_time": 75.48,
          "assigned_machine": "PP09-160T-C-A1"  // Pre-assigned
        }
      ]
    }
  },
  "machines": ["PP09-160T-C-A1", "WH01A-PK", ...]  // 145 machines
}
```

### Constraint Implementation

#### 1. Sequence Constraints (Hard) ✅
- Tasks within family must complete in order (1/3 → 2/3 → 3/3)
- Cannot start sequence N+1 until sequence N is completed
- Tracked via family_completion_status dictionary
- Fixed critical bug: Added sequence_available time checking

#### 2. Machine Assignment (Hard) ✅
- Tasks with `assigned_machine`: Must use that specific machine
- Tasks without `assigned_machine` (6% only): Can use any available machine
- One task per machine at a time
- Machine availability tracked via timeline


### Action Space & Masking
```python
# Action = index of task to schedule next
action_space = gym.spaces.Discrete(n_tasks)

# Valid action criteria:
- Task not already scheduled
- Previous sequence in family completed
- Assigned machine is available (or any machine if unassigned)
- Current time >= material_arrival
```

### Reward Structure
- **On-time completion**: +100
- **Early completion bonus**: +50 * days_early
- **Late penalty**: -100 * days_late
- **Sequence violation**: -500 (should never happen with masking)
- **Utilization bonus**: +10 * machine_utilization_rate

### Training Pipeline

#### Curriculum Learning Stages
```
Stage 1: 10 jobs (34 tasks) - Learn basic sequencing
Stage 2: 20 jobs (65 tasks) - Handle urgency
Stage 3: 40 jobs (130 tasks) - Resource contention
Stage 4: 60 jobs (195 tasks) - Complex dependencies
Stage 5: 100 jobs (327 tasks) - Near production scale
Stage 6: 200+ jobs (600+ tasks) - Full production complexity
```

#### PPO Configuration
- **Network**: MLP (256-128-64 units)
- **Learning rate**: 3e-4
- **Batch size**: 64
- **Entropy coefficient**: 0.01
- **Clip range**: 0.2
- **Training timesteps**: 100k per stage

## Data Processing Pipeline

### 1. Data Extraction from MariaDB
```
tbl_jo_txn + tbl_jo_process → Raw Job Data
              ↓
    Calculate Processing Time:
    If CapMin_d = 1 and CapQty_d > 0:
        hours = (JoQty_d / (CapQty_d * 60)) + SetupTime_d
              ↓
    Extract Machine Assignment:
    Machine_v → assigned_machine_name (via tbl_machine lookup)
              ↓
    Create JSON Snapshot with families and tasks
```

### 2. Environment State Representation
```python
state = {
    'task_ready': [0/1 for each task],      # Can be scheduled?
    'machine_busy': [0/1 for each machine],  # Currently occupied?
    'time_progress': current_time / horizon,  # Normalized time
    'urgency_scores': days_to_lcd / max_lcd, # Deadline pressure
    'sequence_progress': completed / total    # Family completion
}
```

### 3. Action Execution
```python
def step(action):
    task = tasks[action]
    if task.assigned_machine:
        machine = task.assigned_machine
    else:
        machine = find_available_machine()
    
    schedule_task_on_machine(task, machine)
    update_family_status(task.family)
    calculate_reward()
```

## Training Workflow

### Phase 1: Environment Setup
- Create Gym-compatible scheduling environment
- Implement constraint validation
- Build reward calculator
- Test with smallest dataset (10_jobs.json)

### Phase 2: PPO Implementation
- Build policy and value networks
- Implement PPO algorithm with clipping
- Add action masking layer
- Create rollout buffer

### Phase 3: Curriculum Training
- Start with Stage 1 (10 jobs)
- Train 100k timesteps per stage
- Progress when performance > 80%
- Save best model after each stage

### Phase 4: Evaluation
- Test on held-out data
- Compare against FIFO baseline
- Generate Gantt charts
- Calculate metrics (utilization, on-time rate, makespan)

## Deployment Workflow

### 1. Inference Pipeline
```
Load trained model → Receive job batch
                  → Create environment
                  → Run episode to completion
                  → Extract schedule
                  → Return JSON response
```

### 2. API Integration
- FastAPI endpoint: POST /schedule
- Input: Job families with tasks
- Output: Scheduled tasks with start/end times
- Response time: <1 second for 100 jobs

## Performance Monitoring

### Training Metrics
- Episode reward progression
- Constraint violation rate (should be 0%)
- Average makespan
- On-time delivery rate
- Machine utilization

### Production Metrics
- Schedule generation time (<1 second)
- Real on-time delivery rate
- Machine utilization rates
- Comparison with current scheduler

## Key Simplifications in app3

1. **No capable_machines complexity**: Tasks have specific assigned machines
2. **Simplified action space**: Select task, not job-machine pair
3. **Pre-processed data**: JSON snapshots ready for training
4. **Focused constraints**: Only essential scheduling rules
5. **Clean architecture**: Separate concerns (env, model, training, eval)

## Emergency Procedures

### Model Failure
- Log error with full context
- Return error response (no fallback)
- Alert operations team
- NO mock schedulers or degraded mode

### Performance Issues
- Monitor inference time
- Check constraint satisfaction
- Validate input data format
- Consider model retraining if degradation detected

## Phase 3 Implementation Details (Completed)

### Curriculum Training Architecture
The curriculum trainer progressively trains PPO models through 6 stages of increasing complexity:

#### Stage Progression
1. **Stage 1 (Toy Easy)**: 10 jobs, 50k steps, 90% success threshold
2. **Stage 2 (Toy Normal)**: 20 jobs, 100k steps, 85% success threshold  
3. **Stage 3 (Small)**: 40 jobs, 150k steps, 80% success threshold
4. **Stage 4 (Medium)**: 60 jobs, 200k steps, 75% success threshold
5. **Stage 5 (Large)**: 100 jobs, 300k steps, 70% success threshold
6. **Stage 6 (Production)**: 200+ jobs, 500k steps, 65% success threshold

#### Key Features
- **Independent Models**: Each stage creates new PPO model (handles dimension changes)
- **Learning Rate Decay**: LR multiplied by 0.9 for each subsequent stage
- **Performance Gating**: Progression requires meeting success threshold
- **Comprehensive Checkpointing**: Best and final models saved per stage
- **Metrics Tracking**: Tensorboard logging and JSON results export

#### Critical Fixes Applied
- **NaN Handling**: Uniform distribution fallback when all actions masked
- **Batch Masking**: Per-element checking for batch processing
- **Dimension Compatibility**: New models per stage for varying obs/action dims

### Optimized Training Parameters (Updated)

#### Success Criteria
- **Completion Threshold**: 80% task completion counts as success (was 100%)
- **Episode Length**: 1500-2500 steps based on complexity (was 1000)
- **Partial Credit**: Model receives rewards for partial progress

#### Reward Structure (Rebalanced)
- **On-time Completion**: +100
- **Early Bonus**: +50 per day early
- **Late Penalty**: -30 per day (reduced from -100)
- **Sequence Violation**: -100 (reduced from -500)
- **Action Taken**: +15 (increased from +5)
- **Utilization Bonus**: +20 (doubled from +10)
- **Completion Bonus**: +1000 for all tasks scheduled

#### Training Configuration
- **Learning Rate**: 5e-4 with 0.9x decay per stage
- **Batch Size**: 128 (optimized for M4 Pro)
- **Rollout Steps**: 2048
- **Success Thresholds**: 70%→60%→50%→40%→30%→20% (progressive)
- **Training Time**: 75k-500k timesteps per stage

---

*This workflow represents the simplified app3 architecture that leverages pre-assigned machines to make PPO training more tractable while maintaining all essential scheduling constraints.*