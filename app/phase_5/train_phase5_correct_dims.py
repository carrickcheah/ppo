#!/usr/bin/env python3
"""
Train Phase 5 with correct real data dimensions (320 jobs, 145 machines)
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def create_env(rank: int = 0):
    """Create environment with correct dimensions"""
    def _init():
        env = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=320,  # Real data has 320 jobs
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=1000,
            use_break_constraints=True,
            use_holiday_constraints=True,
            invalid_action_penalty=-10.0,
            seed=42 + rank
        )
        return Monitor(env)
    return _init

def train_correct_dims():
    print("\n" + "="*60)
    print("Phase 5 Training with Correct Dimensions")
    print("Real Data: 320 jobs, 145 machines")
    print("="*60 + "\n")
    
    # Create parallel environments
    n_envs = 4
    print(f"Creating {n_envs} training environments...")
    train_envs = SubprocVecEnv([create_env(i) for i in range(n_envs)])
    
    # Check action space
    print(f"Action space: {train_envs.action_space}")
    print(f"Total action combinations: 320 × 145 = {320 * 145:,}")
    print(f"Hierarchical action space: 320 + 145 = 465")
    
    # Create PPO model with adjusted hyperparameters
    print("\nCreating PPO model with optimized hyperparameters...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=0.001,      # Higher for faster learning
        n_steps=2048 // n_envs,
        batch_size=128,           # Smaller batches
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,           # More exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path="models/multidiscrete/correct_dims",
        name_prefix="phase5_320jobs"
    )
    
    # Train
    total_timesteps = 500000  # Start with 500k
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("Expected time: 10-15 minutes\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Save final model
    os.makedirs("models/multidiscrete/correct_dims", exist_ok=True)
    model.save("models/multidiscrete/correct_dims/final_model")
    print("Model saved")
    
    # Test the model
    print("\n" + "="*40)
    print("Testing trained model...")
    print("="*40)
    
    test_env = DummyVecEnv([create_env(0)])
    obs = test_env.reset()
    
    scheduled = 0
    invalid = 0
    
    print("\nRunning test episode (200 steps)...")
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
        
        if step % 50 == 0:
            invalid_rate = invalid/(step+1)*100
            print(f"  Step {step}: {scheduled} jobs scheduled, {invalid_rate:.1f}% invalid")
        
        if done[0]:
            break
    
    print(f"\nFinal Results:")
    print(f"  Jobs scheduled: {scheduled}/320")
    print(f"  Invalid action rate: {invalid/(step+1)*100:.1f}%")
    print(f"  Completion rate: {scheduled/320*100:.1f}%")
    
    if scheduled == 320:
        makespan = info[0].get('makespan', 0)
        print(f"  Makespan: {makespan:.1f} hours")
        print(f"\n✅ SUCCESS! All jobs scheduled!")

if __name__ == "__main__":
    # Ensure we have fresh data
    print("Refreshing real production data...")
    os.system("uv run python src/data_ingestion/ingest_data.py --output data/real_production_snapshot.json")
    
    # Train with correct dimensions
    train_correct_dims()