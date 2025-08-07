# PPO Scheduling System

A production-ready scheduling system using Proximal Policy Optimization (PPO) for intelligent job scheduling in manufacturing environments.

## System Overview

This system leverages deep reinforcement learning to optimize production scheduling, handling complex constraints while maximizing on-time delivery and machine utilization.

## Key Features

- **Real Production Data**: Uses actual job data from MariaDB with JOAW, JOST, JOPRD prefixes
- **PPO-based Learning**: State-of-the-art reinforcement learning for scheduling decisions
- **Web Visualization**: Interactive Gantt charts with Jobs and Machines views
- **Auto-Detection**: Automatic discovery of datasets and trained models
- **Constraint Handling**: Respects sequence dependencies, machine assignments, and material arrival dates
- **Scalable Architecture**: Handles 10 to 500+ job families

## Project Structure

```
ppo/
├── app/                    # Original implementation
├── app_2/                  # Phase-based development
├── app3/                   # Production system
│   ├── api/               # FastAPI backend
│   ├── src/               # Core scheduling logic
│   ├── data/              # JSON datasets
│   ├── checkpoints/       # Trained models
│   ├── visualizations/    # Generated charts
│   ├── phase3/           # Analysis outputs
│   └── frontend3/        # React web app
└── docs/                  # Documentation

```

## Quick Start

### 1. Start the API Server
```bash
cd app3
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd app3/frontend3
npm install
npm run dev
```

### 3. Access the Application
Open http://localhost:5173 in your browser

## Training Models

### Using Stable Baselines3 (Recommended)
```bash
cd app3
uv run python train_sb3_1million.py  # Train 1M timesteps model
uv run python train_sb3_500k.py      # Train 500k timesteps model
uv run python train_sb3_optimized.py # Train with optimized hyperparameters
```

### Models Auto-Detection
Trained models are automatically detected from `app3/checkpoints/` directory. Any model with `best_model.zip` will appear in the UI dropdown.

## Dataset Management

### Available Datasets
- **10_jobs.json**: 34 tasks (training stage 1)
- **20_jobs.json**: 65 tasks (training stage 2)
- **40_jobs.json**: 130 tasks (training stage 3)
- **60_jobs.json**: 195 tasks (training stage 4)
- **100_jobs.json**: 327 tasks (training stage 5)
- **200_jobs.json**: 650+ tasks (training stage 6)
- **500_jobs.json**: 1600+ tasks (production scale)

Datasets are automatically detected from `app3/data/` directory.

## Results Analysis

### Generate Analysis Reports
```bash
cd app3
uv run python analyze_and_visualize.py
```

This generates:
- Log files in `phase3/logs/q_analysis_log_*.txt`
- JSON results in `phase3/results/q_results_*.json`
- Job allocation charts in `visualizations/q_job_allocation_*.png`
- Machine allocation charts in `visualizations/q_machine_allocation_*.png`

## Performance Metrics

### Latest Results (100 jobs, sb3_1million model)
- **Completion Rate**: 100% (327/327 tasks)
- **On-Time Rate**: 29.05%
- **Machine Utilization**: 8.96%
- **Makespan**: 888.93 hours
- **Inference Time**: 7.45 seconds

## API Endpoints

### Schedule Jobs
```bash
POST /api/schedule
{
  "dataset": "100_jobs",
  "model": "sb3_1million",
  "deterministic": true,
  "max_steps": 10000
}
```

### List Datasets
```bash
GET /api/datasets
```

### List Models
```bash
GET /api/models
```

## Visualization Features

### Jobs View
- Each sequence displayed on separate row
- Format: FAMILY_PROCESS_SEQUENCE/TOTAL
- Color coding by deadline status:
  - Green: >72h to deadline
  - Yellow: <72h to deadline
  - Orange: <24h to deadline
  - Red: Late

### Machines View
- Per-machine job allocation
- Utilization percentage display
- Timeline visualization
- Resource conflict detection

## Development

### Environment Setup
```bash
cd app3
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Running Tests
```bash
pytest tests/
```

### Code Standards
- Python 3.12+ required
- Type hints on all functions
- Follow CLAUDE.md guidelines
- No hardcoded scheduling logic
- Real production data only

## Troubleshooting

### Common Issues

1. **API Connection Refused**
   - Ensure API server is running: `uvicorn api.main:app`
   - Check port 8000 is not in use

2. **Model Not Found**
   - Verify model exists in `checkpoints/` directory
   - Check for `best_model.zip` file

3. **Frontend Build Issues**
   - Clear npm cache: `npm cache clean --force`
   - Reinstall dependencies: `rm -rf node_modules && npm install`

## Contributing

1. Follow the workflow in FLOWS.md
2. Update ACTIVITY_LOG.md with changes
3. Mark completed items in TODO.md
4. Maintain code quality standards

## License

Proprietary - All rights reserved

## Contact

For questions or support, refer to project documentation or contact the development team.

---

*Last Updated: 2025-08-07*
*Version: 1.0.0 Production Release*