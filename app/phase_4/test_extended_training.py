#!/usr/bin/env python3
"""
Quick test of extended training - runs for just 1000 steps to verify setup
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environments.full_production_env import FullProductionEnv

print("="*60)
print("PHASE 4 EXTENDED TRAINING - QUICK TEST")
print("="*60)
print("Testing with 1000 steps only\n")

# Create environment
print("1. Creating environment with real data...")
env = DummyVecEnv([lambda: FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)])

# Load model
print("2. Loading model...")
model = PPO.load("models/full_production/final_model.zip", env=env)

# Update hyperparameters
print("3. Setting optimized hyperparameters...")
model.learning_rate = 3e-5
model.batch_size = 1024
model.ent_coef = 0.005

print(f"   Learning rate: {model.learning_rate}")
print(f"   Batch size: {model.batch_size}")
print(f"   Entropy: {model.ent_coef}")

# Quick test
print("\n4. Running 1000 steps test...")
model.learn(
    total_timesteps=1000,
    progress_bar=True,
    reset_num_timesteps=False
)

print("\n✓ Success! Training setup verified.")
print("\nTo run full extended training (500k steps):")
print("  uv run python phase_4/run_extended_training_now.py")

env.close()