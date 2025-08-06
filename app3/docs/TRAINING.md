# PPO Training Guide

## Overview

This guide explains how to train PPO models for production scheduling. The system uses curriculum learning with progressive difficulty stages.

## Prerequisites

- Python 3.12+
- Apple Silicon Mac (for MPS) or NVIDIA GPU (for CUDA)
- At least 8GB RAM
- Real production data in JSON format

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/carrickcheah/Project/ppo/app3
uv sync
```

### 2. Prepare Data

Ensure you have JSON snapshots in the `data/` directory:
- `40_jobs.json` - Small scale (127 tasks)
- `100_jobs.json` - Medium scale (327 tasks)
- `200_jobs.json` - Large scale (650+ tasks)

### 3. Run Training

```bash
# Quick training (40 jobs)
uv run python train_single_stage.py

# Curriculum training (progressive stages)
uv run python src/training/curriculum_trainer.py

# Custom training
uv run python train_100_jobs_simple.py
```

## Training Stages

### Stage 1: Basic (40 jobs)
- **Tasks**: 127
- **Focus**: Learn basic sequencing
- **Timesteps**: 100,000
- **Success Threshold**: 80%

### Stage 2: Medium (100 jobs)
- **Tasks**: 327
- **Focus**: Handle urgency and contention
- **Timesteps**: 200,000
- **Success Threshold**: 70%

### Stage 3: Large (200+ jobs)
- **Tasks**: 650+
- **Focus**: Complex dependencies
- **Timesteps**: 300,000
- **Success Threshold**: 60%

## Hyperparameters

### PPO Settings
```yaml
learning_rate: 3e-4
batch_size: 64
n_epochs: 10
clip_range: 0.2
gamma: 0.99
gae_lambda: 0.95
```

### Network Architecture
```yaml
hidden_sizes: [256, 128, 64]
activation: relu
use_layer_norm: true
```

### Reward Weights
```yaml
on_time_reward: 100
early_bonus_per_day: 50
late_penalty_per_day: -30
sequence_violation: -100
action_taken_bonus: 15
```

## Training Scripts

### 1. Single Stage Training

```python
#!/usr/bin/env python
# train_single_stage.py

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler

# Create environment
env = SchedulingEnv('data/40_jobs.json', max_steps=1500)

# Create model
model = PPOScheduler(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    device='mps'  # or 'cuda' for NVIDIA
)

# Training loop
for episode in range(500):
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, info['action_mask'])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    if episode % 10 == 0:
        print(f"Episode {episode}: {info['tasks_scheduled']}/{info['total_tasks']}")

# Save model
model.save('checkpoints/my_model.pth')
```

### 2. Curriculum Training

```bash
# Full curriculum with all stages
uv run python src/training/curriculum_trainer.py \
  --stages data/40_jobs.json data/100_jobs.json data/200_jobs.json \
  --timesteps 100000 200000 300000 \
  --device mps

# Custom curriculum
uv run python src/training/curriculum_trainer.py \
  --stages data/10_jobs.json data/20_jobs.json \
  --timesteps 50000 100000 \
  --success-threshold 0.8
```

### 3. Fast Training (Optimized)

```bash
# Fast training script for M4 Pro
./run_fast_training.sh

# Or manually
uv run python train_improved.py \
  --batch-size 128 \
  --learning-rate 5e-4 \
  --device mps \
  --timesteps 100000
```

## Monitoring Training

### Tensorboard

```bash
# Start tensorboard
tensorboard --logdir tensorboard/

# View at http://localhost:6006
```

### Metrics to Watch

1. **Completion Rate**: Should increase over time
2. **Average Reward**: Should trend upward
3. **Policy Loss**: Should decrease and stabilize
4. **Value Loss**: Should decrease
5. **Entropy**: Should decrease (policy becoming more deterministic)

## Common Issues and Solutions

### Issue: Training Too Slow

**Solution**: Reduce batch size or use faster device
```python
# Reduce batch size
batch_size = 32  # Instead of 64

# Use MPS on Apple Silicon
device = 'mps'

# Or use CPU with fewer workers
device = 'cpu'
```

### Issue: Model Not Converging

**Solution**: Adjust learning rate or reward weights
```python
# Lower learning rate
learning_rate = 1e-4  # Instead of 3e-4

# Adjust reward weights
rewards = {
    'on_time_reward': 100,
    'late_penalty_per_day': -20,  # Reduce penalty
    'action_taken_bonus': 20  # Increase incentive
}
```

### Issue: Poor Performance on Large Scale

**Solution**: Use curriculum learning
```python
# Start with smaller problems
stages = [
    ('data/10_jobs.json', 50000),
    ('data/20_jobs.json', 100000),
    ('data/40_jobs.json', 150000),
    ('data/100_jobs.json', 200000)
]

for data_path, timesteps in stages:
    train_stage(data_path, timesteps)
```

### Issue: Overfitting

**Solution**: Add regularization
```python
# Increase entropy coefficient
entropy_coef = 0.02  # Instead of 0.01

# Add dropout (modify networks.py)
self.dropout = nn.Dropout(0.1)

# Use early stopping
patience = 50  # Stop if no improvement for 50 episodes
```

## Performance Optimization

### For Apple Silicon (M1/M2/M3/M4)

```python
# Use MPS acceleration
device = 'mps'

# Optimize batch size for unified memory
batch_size = 128  # M4 Pro can handle larger batches

# Use mixed precision (if supported)
torch.set_float32_matmul_precision('medium')
```

### For NVIDIA GPUs

```python
# Use CUDA
device = 'cuda'

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Use larger batch sizes
batch_size = 256
```

### For CPU

```python
# Use CPU with threading
device = 'cpu'
torch.set_num_threads(8)

# Smaller batch size
batch_size = 32
```

## Evaluation

After training, evaluate your model:

```bash
# Evaluate on test data
uv run python src/evaluation/evaluate.py \
  --model checkpoints/best_model.pth \
  --data data/test_jobs.json \
  --episodes 10

# Compare with baselines
uv run python src/evaluation/baselines.py \
  --data data/test_jobs.json \
  --episodes 5
```

## Model Selection

Choose the best model based on:

1. **Completion Rate**: Higher is better (target: >90%)
2. **Constraint Satisfaction**: Must be 100%
3. **On-time Delivery**: Higher is better (target: >85%)
4. **Inference Speed**: <1 second for 100 jobs

## Saving and Loading Models

### Save Model
```python
# Save after training
model.save('checkpoints/my_model.pth')

# Save best model during training
if completion_rate > best_rate:
    model.save('checkpoints/best_model.pth')
    best_rate = completion_rate
```

### Load Model
```python
# Load for inference
model = PPOScheduler(obs_dim, action_dim, device='mps')
model.load('checkpoints/best_model.pth')

# Load for continued training
model.load('checkpoints/checkpoint_ep100.pth')
# Continue training...
```

## Advanced Techniques

### 1. Hyperparameter Tuning

```python
# Grid search
for lr in [1e-4, 3e-4, 5e-4]:
    for batch_size in [32, 64, 128]:
        train_with_params(lr, batch_size)
```

### 2. Ensemble Models

```python
# Train multiple models
models = []
for seed in [42, 123, 456]:
    model = train_with_seed(seed)
    models.append(model)

# Ensemble prediction
predictions = [m.predict(obs) for m in models]
action = majority_vote(predictions)
```

### 3. Transfer Learning

```python
# Load pre-trained model
model.load('checkpoints/40jobs_model.pth')

# Fine-tune on larger dataset
train_on_new_data(model, 'data/100_jobs.json')
```

## Configuration Files

Use YAML configs for reproducibility:

```yaml
# configs/training.yaml
training:
  data_path: "data/40_jobs.json"
  epochs: 500
  batch_size: 64
  learning_rate: 3e-4
  device: "mps"
  
model:
  hidden_sizes: [256, 128, 64]
  activation: "relu"
  
rewards:
  on_time: 100
  late_penalty: -30
```

Load config in training:
```python
import yaml

with open('configs/training.yaml') as f:
    config = yaml.safe_load(f)

model = train_with_config(config)
```

## Tips for Success

1. **Start Small**: Begin with 40-job dataset
2. **Monitor Metrics**: Use tensorboard religiously
3. **Save Checkpoints**: Save models regularly
4. **Test Often**: Evaluate after each training stage
5. **Use Curriculum**: Progressive difficulty works best
6. **Tune Rewards**: Adjust based on business priorities
7. **Validate Constraints**: Ensure 100% constraint satisfaction
8. **Compare Baselines**: Always compare with FIFO/EDD
9. **Document Results**: Keep training logs and metrics
10. **Version Control**: Track model versions and configs