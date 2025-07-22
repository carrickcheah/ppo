#!/usr/bin/env python3
"""
Simple training without evaluation callbacks - just train and test
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def train_simple():
    print("\n" + "="*60)
    print("Phase 5 Simple Training (No Evaluation)")
    print("="*60 + "\n")
    
    # Create single environment
    print("Creating environment...")
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=100,  # Train with 100 jobs
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=200,  # Shorter episodes
        use_break_constraints=False,
        use_holiday_constraints=False,
        invalid_action_penalty=-5.0,
        seed=42
    )
    
    # The environment loads 411 jobs from snapshot but we request 100
    # So action space should be [100, 145] not [411, 145]
    print(f"Actual action space: {env.action_space}")
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.001,
        n_steps=128,  # Very small for quick updates
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        verbose=1
    )
    
    # Train for 20k steps only
    print("\nTraining for 20,000 steps...")
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=20000, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")
    
    # Save model
    model.save("models/multidiscrete/simple_model")
    print("Model saved to: models/multidiscrete/simple_model.zip")
    
    # Test the model
    print("\n" + "="*40)
    print("Testing trained model...")
    print("="*40)
    
    obs = vec_env.reset()
    total_reward = 0
    steps = 0
    scheduled = 0
    invalid = 0
    
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
            
        if done[0]:
            print(f"\nEpisode completed!")
            break
    
    print(f"\nResults:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Jobs scheduled: {scheduled}/100")
    print(f"  Invalid actions: {invalid} ({invalid/steps*100:.1f}%)")
    
    if scheduled == 100:
        makespan = info[0].get('makespan', 0)
        print(f"  Makespan: {makespan:.1f} hours")
        print(f"\n✅ SUCCESS! All jobs scheduled!")
    else:
        print(f"\n⚠️  Only {scheduled}/100 jobs scheduled")
        
    # Test with random policy for comparison
    print("\n" + "="*40)
    print("Random policy baseline:")
    print("="*40)
    
    obs = vec_env.reset()
    random_reward = 0
    random_scheduled = 0
    random_invalid = 0
    
    for _ in range(200):
        action = vec_env.action_space.sample()
        obs, reward, done, info = vec_env.step(action)
        random_reward += reward[0]
        
        if info[0].get('invalid_action', False):
            random_invalid += 1
        else:
            random_scheduled = info[0].get('scheduled_count', 0)
            
        if done[0]:
            break
    
    print(f"  Random scheduled: {random_scheduled}/100")
    print(f"  Random invalid rate: {random_invalid/200*100:.1f}%")
    print(f"  Improvement: {scheduled - random_scheduled} more jobs scheduled")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    train_simple()