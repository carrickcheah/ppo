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
├── data/                    # JSON snapshots (10-500 jobs)
├── database/               # MariaDB connection
├── src/
│   ├── environments/       # Gym environment and constraints
│   ├── models/            # PPO implementation
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Performance metrics
│   ├── visualization/     # Gantt charts and plots
│   └── api/              # FastAPI deployment
├── configs/               # YAML configuration files
├── tests/                # Unit tests
└── visualizations/       # Output charts and graphs
```

## Quick Start

### Prerequisites

- Python 3.12+
- MariaDB connection
- CUDA-capable GPU (optional but recommended)

### Installation

```bash
# Clone repository
cd /Users/carrickcheah/Project/ppo/app3

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add gymnasium stable-baselines3 torch numpy pandas matplotlib pydantic-settings
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
# Start curriculum training
python src/training/curriculum_trainer.py

# Or train specific stage
python src/training/train.py --stage 1 --data data/10_jobs.json
```

### Evaluation

```bash
# Evaluate trained model
python src/evaluation/evaluate.py --model checkpoints/best_model.pth --data data/100_jobs.json

# Generate visualizations
python src/visualization/gantt_chart.py --schedule results/schedule.json
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

## Training Stages

| Stage | Jobs | Tasks | Focus | Timesteps |
|-------|------|-------|-------|-----------|
| 1 | 10 | 34 | Basic sequencing | 100k |
| 2 | 20 | 65 | Urgency handling | 100k |
| 3 | 40 | 130 | Resource contention | 100k |
| 4 | 60 | 195 | Complex dependencies | 100k |
| 5 | 100 | 327 | Near production | 100k |
| 6 | 200+ | 600+ | Full complexity | 100k |

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

## Development

### Running Tests
```bash
pytest tests/
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

## License

Internal use only - Proprietary

## Contact

For questions or issues, please contact the development team.

---

*Following CLAUDE.md guidelines: Real data only, PPO only, no hardcoded scheduling logic*