#!/usr/bin/env python3
"""
Simple Extended Training - Just run it!
Uses the same setup as the original Phase 4 training that worked.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environments.full_production_env import FullProductionEnv


print("="*60)
print("PHASE 4 EXTENDED TRAINING - SIMPLE VERSION")
print("="*60)
print("Current: 49.2h → Target: <45h")
print("-"*60)

# Create directories
os.makedirs("models/full_production/extended", exist_ok=True)
os.makedirs("logs/phase4/extended", exist_ok=True)

# Create environment first (with real data)
print("\n1. Creating environment...")
print("   Using real production data")

env = DummyVecEnv([lambda: FullProductionEnv(
    n_machines=150,  # Use 150 (available real machines)
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)])

# Load the model with the environment
print("\n2. Loading existing model with new environment...")
model = PPO.load("models/full_production/final_model.zip", env=env)

# Update hyperparameters
print("\n3. Optimizing hyperparameters...")
model.learning_rate = 3e-5  # 3x increase
model.batch_size = 1024     # 2x increase  
model.ent_coef = 0.005      # Half

# Simple callback for checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="models/full_production/extended",
    name_prefix="extended"
)

# Train
print("\n4. Starting extended training...")
print("   - Additional 500k steps (about 5 hours)")
print("   - Press Ctrl+C to stop early\n")

try:
    model.learn(
        total_timesteps=500000,  # 500k more steps
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False  # Continue from 1M
    )
    
    print("\n5. Saving final model...")
    model.save("models/full_production/extended/final_extended_model")
    print("\nSuccess! Extended training complete.")
    
except KeyboardInterrupt:
    print("\n\nStopped by user. Saving progress...")
    model.save("models/full_production/extended/interrupted_model")
    print("Saved. Run again to continue.")

finally:
    env.close()

print("\nDone!")