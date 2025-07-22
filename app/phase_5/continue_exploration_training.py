#!/usr/bin/env python3
"""
Continue Phase 5 training from 300k checkpoint
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def create_env(rank: int = 0):
    """Create environment"""
    def _init():
        env = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=320,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=500,
            use_break_constraints=False,
            use_holiday_constraints=False,
            invalid_action_penalty=-5.0,
            seed=42 + rank
        )
        return Monitor(env)
    return _init

def continue_training():
    print("\n" + "="*60)
    print("Continuing Phase 5 Training from 300k Checkpoint")
    print("="*60 + "\n")
    
    # Create environments
    n_envs = 8
    print(f"Creating {n_envs} training environments...")
    train_envs = SubprocVecEnv([create_env(i) for i in range(n_envs)])
    
    # Load model from checkpoint
    print("Loading 300k checkpoint...")
    model = PPO.load(
        "models/multidiscrete/exploration/phase5_explore_300000_steps",
        env=train_envs,
        device='cpu'
    )
    print("Model loaded successfully")
    print(f"Current progress: 15 jobs scheduled (85% invalid)")
    
    # Adjust hyperparameters for continued training
    model.learning_rate = 0.0003  # Slightly lower
    model.ent_coef = 0.08  # Still high but reduced
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path="models/multidiscrete/exploration_continued",
        name_prefix="phase5_explore"
    )
    
    # Continue training
    additional_timesteps = 700_000  # Train to 1M total
    print(f"\nContinuing training for {additional_timesteps:,} timesteps (to 1M total)...")
    print("Expected time: ~15-20 minutes\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,  # Continue from 300k
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    print(f"\nAdditional training time: {train_time/60:.1f} minutes")
    
    # Save final model
    os.makedirs("models/multidiscrete/exploration_continued", exist_ok=True)
    model.save("models/multidiscrete/exploration_continued/model_1m")
    print("1M model saved")
    
    # Quick test
    print("\n" + "="*40)
    print("Testing 1M model...")
    print("="*40)
    
    test_env = DummyVecEnv([create_env(0)])
    obs = test_env.reset()
    
    # Test both deterministic and stochastic
    for deterministic in [False, True]:
        print(f"\n{'Deterministic' if deterministic else 'Stochastic'} mode:")
        obs = test_env.reset()
        
        scheduled = 0
        invalid = 0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = test_env.step(action)
            
            if not info[0].get('invalid_action', True):
                scheduled = info[0].get('scheduled_count', 0)
            else:
                invalid += 1
                
            if step == 49:
                print(f"  After 50 steps: {scheduled} jobs, {invalid/50*100:.1f}% invalid")
        
        print(f"  After 100 steps: {scheduled} jobs, {invalid/100*100:.1f}% invalid")
        
        if scheduled >= 50:
            print("  SUCCESS! Model is learning to schedule jobs effectively")

if __name__ == "__main__":
    continue_training()