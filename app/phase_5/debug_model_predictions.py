#!/usr/bin/env python3
"""
Debug Phase 5 model predictions to understand invalid actions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def debug_predictions():
    print("\n" + "="*60)
    print("Debugging Phase 5 Model Predictions")
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
    
    # Load model
    model = PPO.load("models/multidiscrete/correct_dims/phase5_320jobs_250000_steps")
    
    # Reset and get first observation
    obs = vec_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {vec_env.action_space}")
    
    # Test predictions
    print("\nModel predictions (detailed):")
    for step in range(10):
        # Get model prediction
        action, _ = model.predict(obs, deterministic=True)
        action = action[0]
        
        print(f"\nStep {step}:")
        print(f"  Predicted action: Job {action[0]}, Machine {action[1]}")
        
        # Get the actual environment (wrapped in vec_env)
        actual_env = vec_env.envs[0]
        
        # Check bounds
        if action[0] >= len(actual_env.jobs):
            print(f"  ERROR: Job index {action[0]} out of bounds (max {len(actual_env.jobs)-1})")
        
        if action[1] >= len(actual_env.machines):
            print(f"  ERROR: Machine index {action[1]} out of bounds (max {len(actual_env.machines)-1})")
        
        # Take action
        obs, reward, done, info = vec_env.step([action])
        
        if info[0].get('invalid_action', False):
            print(f"  Invalid: {info[0].get('invalid_reason', 'Unknown')}")
            
            # Check if job is already scheduled
            job_id = action[0]
            if job_id < len(actual_env.jobs):
                job = actual_env.jobs[job_id]
                print(f"  Job {job['job_id']} scheduled: {job_id in actual_env.scheduled_jobs}")
                
                # Check compatibility
                machine_id = action[1]
                if machine_id < len(actual_env.machines):
                    machine = actual_env.machines[machine_id]
                    # Check if machine is in job's capable_machines list
                    capable_machines = job.get('capable_machines', [])
                    compatible = machine['machine_id'] in capable_machines
                    print(f"  Compatible with machine {machine['machine_id']}: {compatible}")
                    if not compatible and capable_machines:
                        print(f"    Job's capable machines: {capable_machines[:5]}...")
        else:
            print(f"  Valid! Reward: {reward[0]:.2f}")
            print(f"  Jobs scheduled: {info[0].get('scheduled_count', 0)}")
    
    # Check what jobs are pending
    actual_env = vec_env.envs[0]
    print("\n" + "-"*40)
    print("Environment state:")
    print(f"  Total jobs: {len(actual_env.jobs)}")
    print(f"  Scheduled jobs: {len(actual_env.scheduled_jobs)}")
    print(f"  Pending jobs: {len([j for j in range(len(actual_env.jobs)) if j not in actual_env.scheduled_jobs])}")
    
    # Sample first few pending jobs
    pending = [j for j in range(len(actual_env.jobs)) if j not in actual_env.scheduled_jobs][:5]
    print(f"\nFirst 5 pending job indices: {pending}")
    
    for job_idx in pending:
        job = actual_env.jobs[job_idx]
        print(f"\n  Job {job_idx} ({job['job_id']}):")
        print(f"    Capable machines: {len(job.get('capable_machines', []))}")
        if job.get('capable_machines'):
            print(f"    First 5: {job['capable_machines'][:5]}")

if __name__ == "__main__":
    debug_predictions()