#!/usr/bin/env python3
"""
Test the 250k checkpoint trained with correct dimensions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_250k_model():
    print("\n" + "="*60)
    print("Testing Phase 5 Model (250k steps, 320 jobs)")
    print("="*60 + "\n")
    
    # Create test environment with correct dimensions
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=123
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load 100k checkpoint
    model = PPO.load("models/multidiscrete/correct_dims/phase5_320jobs_100000_steps")
    print("100k checkpoint loaded successfully")
    print(f"Model expects action space: {model.policy.action_space}")
    
    # Test episode
    obs = vec_env.reset()
    scheduled = 0
    invalid = 0
    total_reward = 0
    
    print("\nRunning test episode...")
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        
        # Careful with the step - it might crash due to missing method
        try:
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            
            if info[0].get('invalid_action', False):
                invalid += 1
            else:
                scheduled = info[0].get('scheduled_count', 0)
            
            if step % 50 == 0:
                print(f"  Step {step}: {scheduled} jobs, {invalid} invalid ({invalid/(step+1)*100:.1f}%)")
                
            if done[0]:
                break
                
        except AttributeError as e:
            print(f"\nError at step {step}: {e}")
            print("The environment has a bug with _calculate_makespan")
            break
    
    print(f"\n250k Model Results:")
    print(f"  Steps taken: {step + 1}")
    print(f"  Jobs scheduled: {scheduled}/320")
    print(f"  Invalid action rate: {invalid/(step+1)*100:.1f}%")
    print(f"  Total reward: {total_reward:.2f}")
    
    # Compare with random baseline
    print("\n" + "-"*40)
    print("Random baseline comparison:")
    
    env_random = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=100,
        seed=456
    )
    vec_env_random = DummyVecEnv([lambda: env_random])
    obs = vec_env_random.reset()
    
    random_scheduled = 0
    random_invalid = 0
    
    for i in range(100):
        action = [vec_env_random.action_space.sample()]
        try:
            obs, reward, done, info = vec_env_random.step(action)
            if info[0].get('invalid_action', False):
                random_invalid += 1
            else:
                random_scheduled = info[0].get('scheduled_count', 0)
        except:
            break
    
    print(f"  Random scheduled: {random_scheduled} jobs in {i+1} steps")
    print(f"  Random invalid rate: {random_invalid/(i+1)*100:.1f}%")
    
    if scheduled > random_scheduled:
        print(f"\n✅ Model is learning! Improvement: +{scheduled - random_scheduled} jobs")
    else:
        print(f"\n⚠️  Model needs more training")

if __name__ == "__main__":
    test_250k_model()