#!/usr/bin/env python3
"""
Test the 550k checkpoint - midway to 1M
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_550k_model():
    print("\n" + "="*60)
    print("Testing 550k Model (Midway Progress)")
    print("="*60 + "\n")
    
    # Create test environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=1000,
        invalid_action_penalty=-5.0,
        seed=42
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    model = PPO.load("models/multidiscrete/exploration_continued/phase5_explore_550000_steps")
    print("550k checkpoint loaded successfully")
    
    # Run full episode
    obs = vec_env.reset()
    
    scheduled = 0
    invalid = 0
    total_reward = 0
    unique_jobs = set()
    
    print("\nRunning full episode (stochastic)...")
    makespan = 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        
        total_reward += reward[0]
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
            # Track unique job selections
            unique_jobs.add(int(action[0][0]))
        
        # Progress updates
        if step % 100 == 0:
            print(f"  Step {step}: {scheduled} jobs, {invalid/(step+1)*100:.1f}% invalid")
            
        if done[0]:
            makespan = info[0].get('makespan', 0)
            break
    
    print(f"\n550k Model Results:")
    print(f"  Jobs scheduled: {scheduled}/320 ({scheduled/320*100:.1f}%)")
    print(f"  Invalid action rate: {invalid/(step+1)*100:.1f}%")
    print(f"  Average reward: {total_reward/(step+1):.2f}")
    print(f"  Unique jobs tried: {len(unique_jobs)}")
    print(f"  Total steps: {step + 1}")
    
    if makespan > 0:
        print(f"  Makespan: {makespan:.1f} hours")
        if makespan < 45:
            print("  ✅ ACHIEVED <45h TARGET!")
    
    # Compare with earlier checkpoints
    print("\n" + "-"*40)
    print("Progress comparison:")
    print("  100k: 95 jobs (90.5% invalid)")
    print("  300k: 98 jobs (90.2% invalid)")
    print(f"  550k: {scheduled} jobs ({invalid/(step+1)*100:.1f}% invalid)")
    
    # Test deterministic for comparison
    print("\n" + "-"*40)
    print("Testing deterministic mode:")
    
    obs = vec_env.reset()
    det_scheduled = 0
    det_invalid = 0
    
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            det_invalid += 1
        else:
            det_scheduled = info[0].get('scheduled_count', 0)
            
        if done[0]:
            break
    
    print(f"  Deterministic: {det_scheduled} jobs, {det_invalid/(step+1)*100:.1f}% invalid")
    
    if scheduled >= 150:
        print("\n✅ Significant progress! Model scheduling ~50% of jobs")
    elif scheduled >= 100:
        print("\n📈 Good progress! Model improving steadily")

if __name__ == "__main__":
    test_550k_model()