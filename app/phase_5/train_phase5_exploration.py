#!/usr/bin/env python3
"""
Train Phase 5 with enhanced exploration and curriculum learning
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

class ExplorationCallback(BaseCallback):
    """Track exploration metrics during training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.unique_actions = set()
        self.valid_actions = 0
        self.total_actions = 0
        
    def _on_step(self) -> bool:
        # Track unique actions
        actions = self.locals['actions']
        for action in actions:
            self.unique_actions.add((int(action[0]), int(action[1])))
        
        # Track validity
        infos = self.locals['infos']
        for info in infos:
            self.total_actions += 1
            if not info.get('invalid_action', True):
                self.valid_actions += 1
        
        # Log every 10k steps
        if self.n_calls % 10000 == 0:
            print(f"\nExploration stats at {self.n_calls} steps:")
            print(f"  Unique actions explored: {len(self.unique_actions)}")
            print(f"  Valid action rate: {self.valid_actions/max(1,self.total_actions)*100:.1f}%")
            
        return True

def create_env(rank: int = 0, start_with_easy_jobs: bool = True):
    """Create environment with optional curriculum learning"""
    def _init():
        env = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=320,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=500,
            use_break_constraints=False,  # Start without constraints
            use_holiday_constraints=False,
            invalid_action_penalty=-5.0,  # Less harsh penalty
            seed=42 + rank
        )
        
        # Curriculum learning: Start with jobs that have many compatible machines
        if start_with_easy_jobs and rank == 0:
            # This would require modifying the environment
            # For now, we'll use the standard environment
            pass
            
        return Monitor(env)
    return _init

def train_with_exploration():
    print("\n" + "="*60)
    print("Phase 5 Training with Enhanced Exploration")
    print("="*60 + "\n")
    
    # Create environments
    n_envs = 8
    print(f"Creating {n_envs} training environments...")
    train_envs = SubprocVecEnv([create_env(i) for i in range(n_envs)])
    
    print(f"Action space: {train_envs.action_space}")
    
    # Create PPO with high exploration
    print("\nCreating PPO model with exploration-focused hyperparameters...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=0.0005,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        ent_coef=0.1,              # High entropy
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=nn.ReLU
        ),
        verbose=1
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path="models/multidiscrete/exploration",
        name_prefix="phase5_explore"
    )
    
    exploration_callback = ExplorationCallback()
    
    callbacks = CallbackList([checkpoint_callback, exploration_callback])
    
    # Train
    total_timesteps = 1_000_000
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("Focus: High exploration to discover valid action patterns\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time/60:.1f} minutes")
    
    # Save final model
    os.makedirs("models/multidiscrete/exploration", exist_ok=True)
    model.save("models/multidiscrete/exploration/final_model")
    
    # Test the model
    print("\n" + "="*40)
    print("Testing trained model...")
    print("="*40)
    
    test_env = DummyVecEnv([create_env(0, start_with_easy_jobs=False)])
    obs = test_env.reset()
    
    valid_count = 0
    total_count = 0
    scheduled = 0
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=False)  # Keep stochastic for variety
        obs, reward, done, info = test_env.step(action)
        
        total_count += 1
        if not info[0].get('invalid_action', True):
            valid_count += 1
            scheduled = info[0].get('scheduled_count', 0)
        
        if step % 20 == 0:
            print(f"  Step {step}: {scheduled} jobs, {valid_count/total_count*100:.1f}% valid")
        
        if done[0]:
            break
    
    print(f"\nFinal Results:")
    print(f"  Jobs scheduled: {scheduled}/320")
    print(f"  Valid action rate: {valid_count/total_count*100:.1f}%")
    print(f"  Unique actions explored during training: {len(exploration_callback.unique_actions)}")

if __name__ == "__main__":
    train_with_exploration()