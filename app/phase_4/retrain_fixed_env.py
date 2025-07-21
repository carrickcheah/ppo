#!/usr/bin/env python3
"""
Retrain with fixed environment that can handle all 411+ jobs
FIXES: max_valid_actions increased to 600 (was 200)
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
print("PHASE 4 RETRAINING - FIXED ENVIRONMENT")
print("="*60)
print("FIX: max_valid_actions increased to 888 (was 200)")
print("This allows scheduling all 411 jobs with plenty of margin")
print("="*60)

# Configuration
total_timesteps = 500_000  # 500k steps
save_freq = 50_000

# Create directories
os.makedirs("models/full_production/fixed", exist_ok=True)

# Create environment with FIXED max_valid_actions
print("\n1. Creating fixed environment...")
env = DummyVecEnv([lambda: FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    max_valid_actions=888,  # FIXED: Was 200, now 888 for safety
    max_episode_steps=3000,  # Also increased for safety
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)])

print("   Environment fixed:")
print("   - max_valid_actions: 888 (can handle all jobs with margin)")
print("   - max_episode_steps: 3000 (extra time to schedule)")

# Create model (can also load previous and continue)
print("\n2. Creating new model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-5,
    n_steps=512,
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

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path="models/full_production/fixed/checkpoints",
    name_prefix="fixed_env_model"
)

# Train
print(f"\n3. Starting training with FIXED environment...")
print(f"   Total timesteps: {total_timesteps:,}")
print(f"   This should achieve proper scheduling of all jobs")
print(f"   Press Ctrl+C to stop early\n")

start_time = datetime.now()

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print("\n4. Training complete! Saving model...")
    model.save("models/full_production/fixed/final_fixed_model")
    
    # Quick test
    print("\n5. Testing fixed environment...")
    obs = env.reset()
    done = False
    steps = 0
    
    # Track scheduling progress
    initial_valid_actions = len(env.envs[0].valid_actions)
    
    while not done and steps < 3000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if steps % 100 == 0:
            valid_actions = len(env.envs[0].valid_actions) if hasattr(env.envs[0], 'valid_actions') else 0
            print(f"   Step {steps}: {valid_actions} valid actions remaining")
    
    # Final results
    if hasattr(env.envs[0], 'completed_tasks'):
        scheduled = sum(len(tasks) for tasks in env.envs[0].completed_tasks.values())
    else:
        scheduled = 0
        
    makespan = info[0].get('makespan', 0)
    print(f"\n   RESULTS WITH FIXED ENVIRONMENT:")
    print(f"   - Jobs scheduled: {scheduled}/411")
    print(f"   - Completion rate: {scheduled/411:.1%}")
    print(f"   - Makespan: {makespan:.1f}h")
    print(f"   - Episode length: {steps} steps")
    
    if scheduled > 400:
        print(f"\n   ✅ SUCCESS! Environment fix worked!")
    else:
        print(f"\n   ⚠️  Still not scheduling all jobs. May need more training.")
    
except KeyboardInterrupt:
    print("\n\nStopped! Saving progress...")
    model.save("models/full_production/fixed/interrupted_fixed_model")

finally:
    training_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nTotal time: {training_time:.1f} minutes")
    env.close()

print("\nNext steps:")
print("1. If all jobs scheduled: Evaluate makespan performance")
print("2. If makespan > 150h: Continue training")
print("3. Once good: Deploy to API")