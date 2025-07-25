# Phase 3: Production Model Training with Curriculum Learning

## Overview

Phase 3 implements a comprehensive 16-stage curriculum learning approach to train a production-ready PPO scheduling model. The curriculum progressively increases complexity from toy problems (5 jobs) to full production scale (500+ jobs).

## Directory Structure

```
phase3/
├── README.md                       # This file
├── start_training.sh              # Training launcher script
├── train_curriculum.py            # Main training script
├── test_model.py                  # Model testing script
├── visualize_training.py          # Progress visualization
├── environments/
│   └── curriculum_env.py          # Curriculum learning environment
├── data_preparation/
│   ├── create_training_snapshots.py    # Create snapshot variations
│   ├── generate_edge_cases.py          # Generate edge case scenarios
│   ├── validate_snapshots.py           # Validate all snapshots
│   └── snapshot_statistics.py          # Generate statistics
├── snapshots/                     # Training data snapshots
│   ├── snapshot_normal.json       # Normal production load
│   ├── snapshot_rush.json         # Rush orders (80% urgent)
│   ├── snapshot_heavy.json        # Heavy load (500+ jobs)
│   ├── snapshot_bottleneck.json   # Machine bottlenecks
│   ├── snapshot_multi_heavy.json  # Multi-machine heavy
│   ├── edge_case_*.json          # Edge case scenarios
│   └── validation_report.md       # Data validation report
├── checkpoints/                   # Model checkpoints (created during training)
├── tensorboard/                   # TensorBoard logs (created during training)
└── visualizations/               # Training visualizations (created by scripts)
```

## Curriculum Stages

### 1. Foundation Training (100k timesteps)
- **Toy Easy** (5 jobs, 3 machines): Learn sequence constraints
- **Toy Normal** (10 jobs, 5 machines): Learn deadline management
- **Toy Hard** (15 jobs, 5 machines): Learn priority handling
- **Toy Multi** (10 jobs, 8 machines): Learn multi-machine coordination

### 2. Strategy Development (200k timesteps)
- **Small Balanced** (30 jobs, 15 machines): Balance multiple objectives
- **Small Rush** (50 jobs, 20 machines): Handle urgent orders
- **Small Bottleneck** (40 jobs, 10 machines): Manage constraints
- **Small Complex** (50 jobs, 25 machines): Complex dependencies

### 3. Scale Training (300k timesteps)
- **Medium Normal** (150 jobs, 40 machines): Scale to medium size
- **Medium Stress** (200 jobs, 50 machines): High load scenarios
- **Large Intro** (300 jobs, 75 machines): Large scale introduction
- **Large Advanced** (400 jobs, 100 machines): Near production scale

### 4. Production Mastery (400k timesteps)
- **Production Warmup** (295 jobs, 145 machines): Real production data
- **Production Rush** (295 jobs, 145 machines): Urgent order scenarios
- **Production Heavy** (500 jobs, 145 machines): Overload handling
- **Production Expert** (500 jobs, 145 machines): Master all scenarios

## Quick Start

### 1. Prepare Data (Already Completed)
```bash
# Generate training snapshots
uv run python phase3/data_preparation/create_snapshots_from_existing.py

# Validate snapshots
uv run python phase3/data_preparation/validate_snapshots.py
```

### 2. Start Training

**Full Curriculum Training** (4-6 hours on M4 Max):
```bash
./phase3/start_training.sh
```

**Resume from Specific Stage**:
```bash
./phase3/start_training.sh --start-stage medium_normal
```

**Train Single Stage Only**:
```bash
./phase3/start_training.sh --single-stage toy_easy
```

### 3. Monitor Progress

**Real-time Monitoring with TensorBoard**:
```bash
tensorboard --logdir phase3/tensorboard
```

**Generate Visualizations**:
```bash
uv run python phase3/visualize_training.py
```

### 4. Test Models

**Test Latest Model**:
```bash
uv run python phase3/test_model.py
```

**Test Specific Stage**:
```bash
uv run python phase3/test_model.py --stage production_expert
```

**Test All Models**:
```bash
uv run python phase3/test_model.py --all
```

## Training on M4 Max

The training is optimized for Apple M4 Max with:
- MPS (Metal Performance Shaders) acceleration
- 8 parallel environments
- Larger batch sizes (up to 4096)
- Progressive learning rate decay

Expected training times:
- Foundation stages: ~15 minutes total
- Strategy stages: ~30 minutes total
- Scale stages: ~45 minutes total
- Production stages: ~60 minutes total
- **Total: 2.5-3 hours**

## Configuration

All training parameters are defined in:
```
configs/phase3_curriculum_config.yaml
```

Key parameters:
- Learning rates: 3e-4 (toy) → 1e-5 (production)
- Batch sizes: 64 (toy) → 4096 (production)
- N-steps: 2048 (toy) → 32768 (production)

## Outputs

### Model Checkpoints
- Best models: `checkpoints/{stage_name}/best_model.zip`
- Final models: `checkpoints/{stage_name}_final.zip`
- Normalization: `checkpoints/{stage_name}_vec_normalize.pkl`

### Training Logs
- TensorBoard logs: `tensorboard/{stage_name}/`
- Training state: `checkpoints/training_state.json`

### Visualizations
- Stage progress: `visualizations/{stage_name}_progress.png`
- Curriculum overview: `visualizations/curriculum_overview.png`
- Training report: `visualizations/training_report.md`

## Next Steps

After training completes:

1. **Evaluate Final Model**:
   ```bash
   uv run python phase3/test_model.py --stage production_expert
   ```

2. **Deploy Model** (Phase 4):
   - Build FastAPI inference server
   - Add working hours filter
   - Integrate with frontend

3. **Fine-tune if Needed**:
   - Adjust reward weights
   - Add more edge cases
   - Extend training on specific stages

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Reduce n_envs (parallel environments)
- Use CPU instead of MPS

### Training Not Converging
- Check reward function weights
- Verify data quality
- Extend training timesteps

### Model Not Improving
- Check if learning rate is too low
- Verify environment observations
- Review reward shaping

## Key Files

- `curriculum_env.py`: Core environment implementation
- `train_curriculum.py`: Training orchestration
- `phase3_curriculum_config.yaml`: All hyperparameters
- `test_model.py`: Performance evaluation