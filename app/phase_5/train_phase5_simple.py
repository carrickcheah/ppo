#!/usr/bin/env python3
"""
Simplified Phase 5 extended training without complex callbacks
Target: Train to 500k timesteps as intermediate goal
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def create_env(rank: int = 0):
    """Create environment for training"""
    def _init():
        env = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=411,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=1000,
            use_break_constraints=True,
            use_holiday_constraints=True,
            invalid_action_penalty=-10.0,
            seed=42 + rank
        )
        return Monitor(env)
    return _init

def train_simple():
    print("\n" + "="*60)
    print("Phase 5 Simplified Extended Training")
    print("Target: 500k timesteps (intermediate goal)")
    print("="*60 + "\n")
    
    # Create parallel environments
    n_envs = 4  # Reduced for stability
    print(f"Creating {n_envs} training environments...")
    train_envs = SubprocVecEnv([create_env(i) for i in range(n_envs)])
    
    # Load checkpoint
    checkpoint_path = "models/multidiscrete/fixed/phase5_fixed_100000_steps.zip"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=train_envs)
    print("Checkpoint loaded - continuing from 100k steps")
    
    # Simple checkpoint callback only
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,  # Every 50k steps
        save_path="models/multidiscrete/simple",
        name_prefix="phase5_simple"
    )
    
    # Train
    target_timesteps = 400000  # Train to 500k total (100k + 400k)
    print(f"\nTraining for {target_timesteps:,} additional timesteps...")
    print("Expected time: 10-15 minutes\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=target_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from 100k
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Save model
    os.makedirs("models/multidiscrete/simple", exist_ok=True)
    model.save("models/multidiscrete/simple/model_500k")
    print("Model saved to: models/multidiscrete/simple/model_500k.zip")
    
    # Quick test
    print("\n" + "="*40)
    print("Testing model...")
    print("="*40)
    
    test_env = DummyVecEnv([create_env(0)])
    obs = test_env.reset()
    
    scheduled = 0
    invalid = 0
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
            
        if step % 100 == 0:
            print(f"Step {step}: {scheduled} jobs scheduled, {invalid} invalid actions")
            
        if done[0]:
            break
    
    print(f"\nFinal: {scheduled}/411 jobs scheduled")
    print(f"Invalid action rate: {invalid/(step+1)*100:.1f}%")
    
    if scheduled == 411:
        makespan = info[0].get('makespan', 0)
        print(f"Makespan: {makespan:.1f} hours")

if __name__ == "__main__":
    train_simple()