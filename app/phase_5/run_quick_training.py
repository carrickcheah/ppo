#!/usr/bin/env python3
"""
Quick training run for Phase 5 to get initial results
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml

# Modify config for quick run
config_path = "configs/phase5_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Reduce timesteps for quick test
config['training']['total_timesteps'] = 50000  # 50k steps
config['training']['n_envs'] = 4  # Fewer environments for stability
config['training']['eval_freq'] = 10000
config['training']['checkpoint_freq'] = 25000

# Save modified config
quick_config_path = "configs/phase5_quick_config.yaml"
with open(quick_config_path, 'w') as f:
    yaml.dump(config, f)

print("Created quick config. Running training...")

# Run training with quick config
import subprocess
result = subprocess.run([
    "uv", "run", "python", "phase_5/train_multidiscrete_ppo.py",
    "--config", quick_config_path
], capture_output=False)

print(f"\nTraining completed with exit code: {result.returncode}")