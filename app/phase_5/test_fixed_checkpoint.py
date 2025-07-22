#!/usr/bin/env python3
"""
Test the fixed Phase 5 checkpoint
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_checkpoint():
    print("\n" + "="*60)
    print("Testing Phase 5 Fixed Model (100k steps)")
    print("="*60 + "\n")
    
    # Create test environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=123
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load checkpoint
    model = PPO.load("models/multidiscrete/fixed/phase5_fixed_100000_steps")
    print("Model loaded successfully")
    
    # Test episode
    obs = vec_env.reset()
    total_reward = 0
    steps = 0
    scheduled = 0
    invalid = 0
    
    print("\nRunning test episode...")
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
        
        if steps % 50 == 0:
            print(f"  Step {steps}: {scheduled} jobs, {invalid} invalid")
            
        if done[0]:
            break
    
    print(f"\nResults after 100k training steps:")
    print(f"  Jobs scheduled: {scheduled}/411")
    print(f"  Invalid action rate: {invalid/steps*100:.1f}%")
    print(f"  Total reward: {total_reward:.2f}")
    
    if scheduled > 0:
        print(f"\n✅ Model is learning! Scheduled {scheduled} jobs")
        if scheduled == 411:
            makespan = info[0].get('makespan', 0)
            print(f"  Makespan: {makespan:.1f} hours")
    else:
        print(f"\n⚠️  Model needs more training")
    
    # Compare with random
    print("\n" + "-"*40)
    print("Random baseline comparison:")
    
    env.reset()
    vec_env = DummyVecEnv([lambda: env])
    obs = vec_env.reset()
    
    random_scheduled = 0
    random_invalid = 0
    
    for _ in range(100):
        action = [vec_env.action_space.sample()]
        obs, reward, done, info = vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            random_invalid += 1
        else:
            random_scheduled = info[0].get('scheduled_count', 0)
    
    print(f"  Random scheduled: {random_scheduled} jobs")
    print(f"  Random invalid rate: {random_invalid}%")
    print(f"\n  Trained model improvement: +{scheduled - random_scheduled} jobs")

if __name__ == "__main__":
    test_checkpoint()