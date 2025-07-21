#!/usr/bin/env python3
"""
Quick test of retraining setup - 10k steps only
To verify everything works before full training
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environments.full_production_env import FullProductionEnv
import torch.nn as nn

print("QUICK RETRAINING TEST (10k steps)")
print("="*40)

# Single environment for quick test
env = DummyVecEnv([lambda: FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)])

print("1. Environment created")
print(f"   Observation space: {env.observation_space}")
print(f"   Action space: {env.action_space}")

# Create model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-5,
    n_steps=256,
    batch_size=64,  # Smaller for quick test
    n_epochs=4,
    gamma=0.995,
    policy_kwargs={
        "net_arch": [512, 512, 256],
        "activation_fn": nn.Tanh
    },
    verbose=1
)

print("\n2. Model created")

# Quick training
print("\n3. Running 10k steps...")
model.learn(total_timesteps=10000, progress_bar=True)

# Test inference
print("\n4. Testing inference...")
obs = env.reset()
for i in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"   Step {i+1}: Action={action[0]}, Reward={reward[0]:.1f}")

env.close()
print("\n✓ Test complete! Ready for full training.")
print("\nRun: uv run python phase_4/retrain_real_production.py")