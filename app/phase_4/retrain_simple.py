#!/usr/bin/env python3
"""
Simple retraining script without multiprocessing
Trains on single environment to avoid multiprocessing issues
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environments.full_production_env import FullProductionEnv
import torch.nn as nn
from datetime import datetime

print("="*60)
print("PHASE 4 RETRAINING - SIMPLE VERSION")
print("="*60)
print("Target: 130h makespan")
print("Single environment (slower but stable)")
print("="*60)

# Configuration
total_timesteps = 500_000  # 500k for faster initial results
save_freq = 50_000

# Create directories
os.makedirs("models/full_production/real_data", exist_ok=True)
os.makedirs("logs/phase4/real_data", exist_ok=True)

# Create single environment
print("\n1. Creating environment...")
env = DummyVecEnv([lambda: FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)])

print("   Environment ready")
print(f"   Workload: 14,951h across 149 machines")
print(f"   Theoretical minimum: 100.3h")

# Create model
print("\n2. Creating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-5,
    n_steps=512,  # More steps since single env
    batch_size=512,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.005,
    policy_kwargs={
        "net_arch": [512, 512, 256],
        "activation_fn": nn.Tanh
    },
    verbose=1,
    device="auto"
)

print("   Model created with optimized hyperparameters")

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path="models/full_production/real_data/checkpoints",
    name_prefix="real_simple"
)

# Train
print(f"\n3. Starting training...")
print(f"   Total timesteps: {total_timesteps:,}")
print(f"   Estimated time: 2-3 hours")
print(f"   Press Ctrl+C to stop early\n")

start_time = datetime.now()

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print("\n4. Training complete! Saving model...")
    model.save("models/full_production/real_data/simple_real_model")
    
    # Quick test
    print("\n5. Quick evaluation...")
    obs = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps += 1
        
    makespan = info[0].get('makespan', 0)
    print(f"\n   Test makespan: {makespan:.1f}h")
    print(f"   vs. Target: 130h")
    print(f"   vs. Minimum: 100.3h")
    
except KeyboardInterrupt:
    print("\n\nStopped! Saving progress...")
    model.save("models/full_production/real_data/simple_interrupted")
    print("Saved.")

finally:
    training_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nTotal time: {training_time:.1f} minutes")
    env.close()

print("\nDone!")
print("\nTo continue training:")
print("- Load the saved model and train for more steps")
print("- Or run the full multiprocess version once working")