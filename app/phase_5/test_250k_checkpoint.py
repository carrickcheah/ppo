#!/usr/bin/env python3
"""
Test the 250k checkpoint to see progress
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_checkpoint():
    print("\n" + "="*60)
    print("Testing Phase 5 Model at 250k steps")
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
    
    # Load 250k checkpoint
    model = PPO.load("models/multidiscrete/simple/phase5_simple_250000_steps")
    print("250k checkpoint loaded")
    
    # Test episode
    obs = vec_env.reset()
    scheduled = 0
    invalid = 0
    
    print("\nRunning test episode...")
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
        
        if step % 50 == 0:
            print(f"  Step {step}: {scheduled} jobs, {invalid} invalid")
            
        if done[0]:
            break
    
    print(f"\n250k Model Results:")
    print(f"  Jobs scheduled: {scheduled}/411")
    print(f"  Invalid action rate: {invalid/(step+1)*100:.1f}%")
    
    if scheduled == 411:
        makespan = info[0].get('makespan', 0)
        print(f"  Makespan: {makespan:.1f} hours")
        print(f"\nComparison:")
        print(f"  Phase 4: 49.2 hours")
        print(f"  Current: {makespan:.1f} hours")
        improvement = (49.2 - makespan) / 49.2 * 100
        print(f"  Improvement: {improvement:.1f}%")
    
    # Compare with 100k
    print("\n" + "-"*40)
    print("Progress from 100k to 250k:")
    print("  100k: 1 job scheduled, 99.8% invalid")
    print(f"  250k: {scheduled} jobs, {invalid/(step+1)*100:.1f}% invalid")

if __name__ == "__main__":
    test_checkpoint()