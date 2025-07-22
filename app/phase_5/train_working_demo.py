#!/usr/bin/env python3
"""
Working demo that properly handles job count
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

def train_working_demo():
    print("\n" + "="*60)
    print("Phase 5 Working Demo")
    print("="*60 + "\n")
    
    # Create environment - let it use all 411 jobs from snapshot
    print("Creating environment with real data...")
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,  # Use all jobs from snapshot
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,  # Enough steps to schedule all
        use_break_constraints=False,
        use_holiday_constraints=False,
        invalid_action_penalty=-5.0,
        seed=42
    )
    
    print(f"Environment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Jobs to schedule: {len(env.jobs) if hasattr(env, 'jobs') else 'unknown'}")
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create model with adjusted hyperparameters
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.0003,  # Lower LR for stability
        n_steps=2048,  # Standard PPO
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Train for 50k steps
    print("\nTraining for 50,000 steps...")
    print("This should take 2-5 minutes\n")
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=50000, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Save model
    os.makedirs("models/multidiscrete/working", exist_ok=True)
    model.save("models/multidiscrete/working/demo_model")
    print("Model saved")
    
    # Test the trained model
    print("\n" + "="*40)
    print("Testing trained model...")
    print("="*40)
    
    # Fresh environment for testing
    test_env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=123
    )
    test_vec_env = DummyVecEnv([lambda: test_env])
    
    obs = test_vec_env.reset()
    total_reward = 0
    steps = 0
    scheduled = 0
    invalid = 0
    
    while steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_vec_env.step(action)
        total_reward += reward[0]
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid += 1
        
        scheduled = info[0].get('scheduled_count', 0)
        
        # Progress update
        if steps % 100 == 0:
            print(f"  Step {steps}: {scheduled} jobs scheduled, {invalid} invalid actions")
            
        if done[0]:
            print(f"\nEpisode completed!")
            break
    
    print(f"\nFinal Results:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Jobs scheduled: {scheduled}/411")
    print(f"  Invalid action rate: {invalid/steps*100:.1f}%")
    
    if scheduled > 0:
        if 'makespan' in info[0]:
            makespan = info[0]['makespan']
            print(f"  Makespan: {makespan:.1f} hours")
        
        if scheduled == 411:
            print(f"\n✅ SUCCESS! All jobs scheduled!")
            print(f"Compare to Phase 4: 49.2 hours")
        else:
            print(f"\n⚠️  Partial success: {scheduled/411*100:.1f}% jobs scheduled")
    
    # Quick random baseline
    print("\n" + "="*40)
    print("Random baseline (first 100 steps):")
    print("="*40)
    
    test_env.reset()
    test_vec_env = DummyVecEnv([lambda: test_env])
    obs = test_vec_env.reset()
    
    random_scheduled = 0
    random_invalid = 0
    
    for _ in range(100):
        # Properly sample from vec env action space
        action = [test_vec_env.action_space.sample()]
        obs, reward, done, info = test_vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            random_invalid += 1
        else:
            random_scheduled = info[0].get('scheduled_count', 0)
    
    print(f"  Random scheduled: {random_scheduled} jobs")
    print(f"  Random invalid rate: {random_invalid}%")
    print(f"  Trained model advantage: {scheduled - random_scheduled} more jobs")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("Next: Run full 2M timestep training for best results")
    print("="*60)

if __name__ == "__main__":
    train_working_demo()