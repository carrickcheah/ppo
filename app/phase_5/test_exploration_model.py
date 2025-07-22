#!/usr/bin/env python3
"""
Test the exploration-focused model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_exploration_model():
    print("\n" + "="*60)
    print("Testing Exploration Model (300k steps)")
    print("="*60 + "\n")
    
    # Create test environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=123
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    model = PPO.load("models/multidiscrete/exploration/phase5_explore_300000_steps")
    print("300k checkpoint loaded")
    
    # Test with deterministic and stochastic predictions
    for deterministic in [True, False]:
        print(f"\n{'Deterministic' if deterministic else 'Stochastic'} predictions:")
        
        obs = vec_env.reset()
        scheduled = 0
        invalid = 0
        unique_actions = set()
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=deterministic)
            unique_actions.add((int(action[0][0]), int(action[0][1])))
            
            obs, reward, done, info = vec_env.step(action)
            
            if info[0].get('invalid_action', False):
                invalid += 1
            else:
                scheduled = info[0].get('scheduled_count', 0)
            
            if step % 20 == 0:
                print(f"  Step {step}: {scheduled} jobs, {invalid/(step+1)*100:.1f}% invalid")
                
            if done[0]:
                break
        
        print(f"\nResults:")
        print(f"  Jobs scheduled: {scheduled}/320")
        print(f"  Invalid action rate: {invalid/(step+1)*100:.1f}%")
        print(f"  Unique actions tried: {len(unique_actions)}")
    
    # Test earlier checkpoint for comparison
    print("\n" + "-"*40)
    print("Testing 100k checkpoint for comparison:")
    
    model_100k = PPO.load("models/multidiscrete/exploration/phase5_explore_100000_steps")
    obs = vec_env.reset()
    
    scheduled = 0
    invalid = 0
    
    for step in range(50):
        action, _ = model_100k.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
    
    print(f"  100k model: {scheduled} jobs, {invalid/50*100:.1f}% invalid")
    
    # Show sample actions
    print("\n" + "-"*40)
    print("Sample actions from 300k model:")
    
    obs = vec_env.reset()
    actual_env = vec_env.envs[0]
    
    for i in range(5):
        action, _ = model.predict(obs, deterministic=False)
        job_idx = int(action[0][0])
        machine_idx = int(action[0][1])
        
        print(f"\nAction {i}: Job {job_idx} → Machine {machine_idx}")
        
        if job_idx < len(actual_env.jobs):
            job = actual_env.jobs[job_idx]
            print(f"  Job ID: {job['job_id']}")
            capable = job.get('capable_machines', [])
            
            if machine_idx < len(actual_env.machines):
                machine = actual_env.machines[machine_idx]
                print(f"  Machine ID: {machine['machine_id']}")
                print(f"  Compatible: {machine['machine_id'] in capable}")

if __name__ == "__main__":
    test_exploration_model()