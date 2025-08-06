# app3 - Simplified PPO Scheduling System

## Overview

A simplified PPO-based production scheduling system that learns to select which task to schedule next while respecting sequence constraints and machine availability. This system leverages pre-assigned machines from real production data to simplify the action space and improve training efficiency.

## Key Features

- **Simplified Action Space**: Select task to schedule next (not job-machine pairs)
- **Pre-assigned Machines**: 94% of tasks have specific machine assignments
- **Real Production Data**: All training data from MariaDB with real job IDs
- **Curriculum Learning**: Progressive training from 10 to 500+ jobs
- **Clean Constraints**: Sequence, availability, and material arrival only

## Project Structure

```
app3/
├── data/                      # Real production JSON snapshots
│   ├── 10_jobs.json          # Stage 1: 34 tasks, 10 families
│   ├── 20_jobs.json          # Stage 2: 65 tasks, 20 families
│   ├── 40_jobs.json          # Stage 3: 130 tasks, 40 families
│   ├── 60_jobs.json          # Stage 4: 195 tasks, 60 families
│   ├── 100_jobs.json         # Stage 5: 327 tasks, 100 families
│   └── 200_jobs.json         # Stage 6: 650+ tasks, 200 families
│
├── src/
│   ├── data/
│   │   └── snapshot_loader.py    # Load and parse JSON data
│   │
│   ├── environments/
│   │   ├── scheduling_env.py     # Gym-compatible environment
│   │   ├── constraint_validator.py # Validate actions and masks
│   │   └── reward_calculator.py  # Calculate rewards
│   │
│   ├── models/
│   │   ├── networks.py          # PolicyValueNetwork with masking
│   │   ├── ppo_scheduler.py     # PPO algorithm implementation
│   │   └── rollout_buffer.py    # Experience storage and GAE
│   │
│   └── training/
│       ├── train.py              # Main training loop
│       └── curriculum_trainer.py # 6-stage curriculum learning
│
├── checkpoints/              # Model checkpoints
├── tensorboard/              # Training logs
└── pyproject.toml           # Dependencies
```

## Quick Start

### Prerequisites

- Python 3.12+
- MariaDB connection
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
cd /Users/carrickcheah/Project/ppo/app3
uv sync
```

### Configuration

1. Set up database credentials in `.env`:
```env
MARIADB_HOST=localhost
MARIADB_USERNAME=myuser
MARIADB_PASSWORD=mypassword
MARIADB_DATABASE=nex_valiant
MARIADB_PORT=3306
```

2. Configure training parameters in `configs/training.yaml`

### Training

```bash
# Full curriculum training (6 stages) - Optimized for M4 Pro
uv run python src/training/curriculum_trainer.py \
  --device mps \
  --batch-size 128 \
  --n-steps 2048 \
  --lr 5e-4

# Quick test run
uv run python src/training/curriculum_trainer.py \
  --stages data/10_jobs.json \
  --device mps \
  --n-steps 1024 \
  --batch-size 64

# Custom stages
uv run python src/training/curriculum_trainer.py --stages data/10_jobs.json data/20_jobs.json
```

### Monitor Training

```bash
tensorboard --logdir tensorboard/curriculum
# View at http://localhost:6006
```

## Data Format

Each JSON snapshot contains:
```json
{
  "families": {
    "JOST25060084": {
      "lcd_date": "2025-08-06",
      "tasks": [
        {
          "sequence": 1,
          "process_name": "CP08-056-1/2",
          "processing_time": 75.48,
          "assigned_machine": "PP09-160T-C-A1"
        }
      ]
    }
  },
  "machines": ["PP09-160T-C-A1", "WH01A-PK", ...]
}
```

## Constraints

### Hard Constraints (Must be satisfied)
1. **Sequence**: Tasks within family complete in order (1/3 → 2/3 → 3/3)
2. **Machine Assignment**: Use pre-assigned machine or any available
3. **No Overlap**: One task per machine at a time
4. **Material Arrival**: Cannot schedule before material date

### Soft Constraints (Learned through rewards)
- Meet LCD deadlines
- Maximize machine utilization
- Minimize makespan
- Prioritize urgent jobs

## Training Stages (Optimized)

| Stage | Jobs | Tasks | Timesteps | Success Threshold | Episode Steps | Focus |
|-------|------|-------|-----------|-------------------|---------------|-------|
| 1 | 10 | 34 | 75k | 70% | 1500 | Basic sequencing |
| 2 | 20 | 65 | 150k | 60% | 1500 | Urgency handling |
| 3 | 40 | 130 | 200k | 50% | 1500 | Resource contention |
| 4 | 60 | 195 | 250k | 40% | 2500 | Complex dependencies |
| 5 | 100 | 327 | 350k | 30% | 2500 | Near production scale |
| 6 | 200+ | 650+ | 500k | 20% | 2500 | Full production |

## Performance Targets

- 95% constraint satisfaction rate
- 85% on-time delivery rate
- <1 second inference for 100 jobs
- >60% machine utilization
- 20% improvement over FIFO baseline

## API Usage

Once trained, deploy the model via FastAPI:

```bash
# Start API server
uvicorn src.api.scheduler_api:app --reload

# Make scheduling request
curl -X POST http://localhost:8000/schedule \
  -H "Content-Type: application/json" \
  -d @data/100_jobs.json
```

## Implementation Status

### Completed (Phases 1-3)
- ✅ Environment with constraint validation and action masking
- ✅ PPO model with clipped objective and GAE
- ✅ Curriculum training pipeline with 6 stages
- ✅ Data loading from real production JSON snapshots
- ✅ Reward calculation with configurable weights
- ✅ Tensorboard integration for monitoring
- ✅ Model checkpointing (best + final per stage)

### Technical Improvements
- Fixed NaN issues with uniform distribution fallback for fully masked states
- Handled batch processing with per-element mask checking
- Resolved dimension mismatches by creating new models per stage
- Implemented learning rate decay (0.9x per stage)
- Optimized reward structure: reduced penalties by 70-80%, increased action incentives by 3x
- Adjusted success criteria: 80% task completion now counts as success
- Extended episode lengths: 1500-2500 steps based on problem complexity
- Tuned for Apple M4 Pro: 200+ iterations/second with MPS acceleration

### Pending (Phases 4-6)
- ⏳ Evaluation metrics and baseline comparisons
- ⏳ Gantt chart visualization
- ⏳ YAML configuration files
- ⏳ FastAPI deployment
- ⏳ Docker containerization

## Development

### Running Tests
```bash
# Test environment
uv run python test_environment.py

# Test PPO model
uv run python test_ppo_model.py
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Type checking
uv run pyright
```

## Documentation

- [TODO.md](TODO.md) - Implementation checklist
- [FLOWS.md](../FLOWS.md) - System workflow documentation
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [ACTIVITY_LOG.md](../ACTIVITY_LOG.md) - Development history

## License

Internal use only - Proprietary

---

*Last Updated: 2025-08-06*
*Following CLAUDE.md guidelines: Real data only, PPO only, no hardcoded scheduling logic*