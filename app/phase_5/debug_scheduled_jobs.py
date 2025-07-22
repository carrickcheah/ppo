#!/usr/bin/env python3
"""
Debug scheduled_jobs tracking issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def debug_scheduled_jobs():
    print("\n" + "="*60)
    print("Debugging scheduled_jobs tracking")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=42
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Reset
    obs = vec_env.reset()
    actual_env = vec_env.envs[0]
    
    print(f"After reset:")
    print(f"  scheduled_jobs type: {type(actual_env.scheduled_jobs)}")
    print(f"  scheduled_jobs length: {len(actual_env.scheduled_jobs)}")
    print(f"  scheduled_count: {actual_env.scheduled_count}")
    
    # Count actual scheduled jobs
    actually_scheduled = sum(1 for s in actual_env.scheduled_jobs if s is not None)
    print(f"  Actually scheduled: {actually_scheduled}")
    
    # Take a random valid action
    print("\nFinding a valid action...")
    for attempt in range(100):
        action = vec_env.action_space.sample()
        job_idx = action[0]
        machine_idx = action[1]
        
        # Check if valid
        if job_idx < len(actual_env.jobs) and machine_idx < len(actual_env.machines):
            job = actual_env.jobs[job_idx]
            machine = actual_env.machines[machine_idx]
            
            # Check if already scheduled
            if actual_env.scheduled_jobs[job_idx] is None:
                # Check compatibility
                capable_machines = job.get('capable_machines', [])
                if machine['machine_id'] in capable_machines:
                    print(f"\nFound valid action: Job {job_idx} → Machine {machine_idx}")
                    print(f"  Job ID: {job['job_id']}")
                    print(f"  Machine ID: {machine['machine_id']}")
                    
                    # Take the action
                    obs, reward, done, info = vec_env.step([action])
                    
                    print(f"\nAfter step:")
                    print(f"  Reward: {reward[0]:.2f}")
                    print(f"  Invalid: {info[0].get('invalid_action', False)}")
                    print(f"  scheduled_count: {actual_env.scheduled_count}")
                    
                    actually_scheduled = sum(1 for s in actual_env.scheduled_jobs if s is not None)
                    print(f"  Actually scheduled: {actually_scheduled}")
                    
                    break
    
    # Test model predictions
    print("\n" + "-"*40)
    print("Testing trained model:")
    
    model = PPO.load("models/multidiscrete/correct_dims/phase5_320jobs_250000_steps")
    
    # Reset for model test
    obs = vec_env.reset()
    actual_env = vec_env.envs[0]
    
    for step in range(5):
        action, _ = model.predict(obs, deterministic=True)
        job_idx = action[0][0]
        machine_idx = action[0][1]
        
        print(f"\nStep {step}: Job {job_idx} → Machine {machine_idx}")
        
        # Check why it might be invalid
        if job_idx < len(actual_env.jobs):
            job = actual_env.jobs[job_idx]
            if actual_env.scheduled_jobs[job_idx] is not None:
                print(f"  Already scheduled!")
            else:
                capable = job.get('capable_machines', [])
                if machine_idx < len(actual_env.machines):
                    machine = actual_env.machines[machine_idx]
                    if machine['machine_id'] not in capable:
                        print(f"  Incompatible! Job needs machines: {capable[:5]}...")
                        print(f"  Selected machine ID: {machine['machine_id']}")
        
        obs, reward, done, info = vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            print(f"  Invalid: {info[0].get('invalid_reason', 'Unknown')}")

if __name__ == "__main__":
    debug_scheduled_jobs()