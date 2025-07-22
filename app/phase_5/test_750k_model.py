#!/usr/bin/env python3
"""
Test the 750k checkpoint - best available model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_750k_model():
    print("\n" + "="*60)
    print("Testing 750k Model (Best Available)")
    print("="*60 + "\n")
    
    # Create test environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=2000,  # Allow longer episode
        invalid_action_penalty=-5.0,
        seed=42
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    model = PPO.load("models/multidiscrete/exploration_continued/phase5_explore_750000_steps")
    print("750k checkpoint loaded successfully")
    
    # Run full episode
    obs = vec_env.reset()
    
    scheduled = 0
    invalid = 0
    total_reward = 0
    unique_jobs = set()
    job_history = []
    
    print("\nRunning full episode until all jobs scheduled or max steps...")
    makespan = 0
    
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        
        total_reward += reward[0]
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
            job_idx = int(action[0][0])
            unique_jobs.add(job_idx)
            job_history.append(job_idx)
        
        # Progress updates
        if step % 200 == 0:
            print(f"  Step {step}: {scheduled} jobs, {invalid/(step+1)*100:.1f}% invalid")
            
        if done[0]:
            makespan = info[0].get('makespan', 0)
            utilization = info[0].get('avg_utilization', 0)
            print(f"\n  Episode completed! All jobs scheduled.")
            break
            
        # Early stop if no progress
        if step > 500 and scheduled < 50:
            print(f"\n  Stopping early - slow progress")
            break
    
    print(f"\n750k Model Results:")
    print(f"  Jobs scheduled: {scheduled}/320 ({scheduled/320*100:.1f}%)")
    print(f"  Invalid action rate: {invalid/(step+1)*100:.1f}%")
    print(f"  Average reward: {total_reward/(step+1):.2f}")
    print(f"  Unique jobs tried: {len(unique_jobs)}")
    print(f"  Total steps: {step + 1}")
    
    if makespan > 0:
        print(f"  Makespan: {makespan:.1f} hours")
        print(f"  Utilization: {utilization:.1f}%")
        if makespan < 45:
            print("  ✅ ACHIEVED <45h TARGET!")
        elif makespan < 50:
            print("  📈 Close to target! (<50h)")
    
    # Show job selection pattern
    if len(job_history) > 10:
        print(f"\n  First 10 scheduled jobs: {job_history[:10]}")
        print(f"  Last 10 scheduled jobs: {job_history[-10:]}")
    
    # Progress summary
    print("\n" + "="*40)
    print("Phase 5 Progress Summary:")
    print("  Random baseline: ~47 jobs (90.6% invalid)")
    print("  100k model: 95 jobs (90.5% invalid)")
    print("  300k model: 98 jobs (90.2% invalid)")
    print("  550k model: 85 jobs (91.5% invalid)")
    print(f"  750k model: {scheduled} jobs ({invalid/(step+1)*100:.1f}% invalid)")
    
    if scheduled >= 200:
        print("\n✅ BREAKTHROUGH! Model scheduling >60% of jobs")
    elif scheduled >= 150:
        print("\n✅ Significant progress! Model scheduling ~50% of jobs")
    elif scheduled >= 100:
        print("\n📈 Good progress! Model improving steadily")
    else:
        print("\n⚠️  Model needs more training or different approach")

if __name__ == "__main__":
    test_750k_model()