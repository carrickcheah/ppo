#!/usr/bin/env python3
"""
Debug why hierarchical model produces 100% invalid actions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def debug_action_space():
    print("\n" + "="*60)
    print("Debugging Hierarchical Action Space")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=100,
        seed=42
    )
    
    # Reset and check initial state
    obs, info = env.reset()
    print(f"Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Observation shape: {obs.shape}")
    print(f"Number of jobs: {len(env.jobs) if hasattr(env, 'jobs') else 'unknown'}")
    print(f"Number of machines: {len(env.machines) if hasattr(env, 'machines') else 'unknown'}")
    
    # Check compatibility matrix
    if hasattr(env, '_compatibility_matrix'):
        compat = env._compatibility_matrix
        print(f"\nCompatibility matrix shape: {compat.shape}")
        print(f"Total compatible pairs: {np.sum(compat)}")
        print(f"Average compatible machines per job: {np.sum(compat) / len(env.jobs):.1f}")
        
        # Check a few specific jobs
        print("\nChecking first 5 jobs:")
        for i in range(min(5, len(env.jobs))):
            job = env.jobs[i]
            compatible_count = np.sum(compat[i])
            print(f"  Job {i} ({job['job_id']}): {compatible_count} compatible machines")
            if compatible_count > 0:
                compatible_machines = np.where(compat[i])[0][:5]  # First 5
                print(f"    Compatible with machines: {compatible_machines.tolist()}")
    
    # Test random actions
    print("\n" + "-"*40)
    print("Testing random actions:")
    invalid_count = 0
    valid_actions = []
    
    for i in range(100):
        action = env.action_space.sample()
        job_idx = action[0]
        machine_idx = action[1]
        
        # Check if this is valid
        if hasattr(env, '_compatibility_matrix'):
            is_valid = env._compatibility_matrix[job_idx, machine_idx]
            if not is_valid:
                invalid_count += 1
            else:
                valid_actions.append((job_idx, machine_idx))
    
    print(f"Random sampling: {invalid_count}/100 invalid ({invalid_count}%)")
    if valid_actions:
        print(f"Example valid actions: {valid_actions[:5]}")
    
    # Test specific known valid actions
    print("\n" + "-"*40)
    print("Testing known valid actions:")
    
    # Find a job with compatible machines
    valid_job_idx = None
    valid_machine_idx = None
    
    if hasattr(env, '_compatibility_matrix'):
        for job_idx in range(len(env.jobs)):
            compatible_machines = np.where(env._compatibility_matrix[job_idx])[0]
            if len(compatible_machines) > 0:
                valid_job_idx = job_idx
                valid_machine_idx = compatible_machines[0]
                break
    
    if valid_job_idx is not None:
        print(f"Found valid pair: Job {valid_job_idx} → Machine {valid_machine_idx}")
        
        # Test this action
        action = np.array([valid_job_idx, valid_machine_idx])
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step result:")
        print(f"  Reward: {reward}")
        print(f"  Invalid action: {info.get('invalid_action', False)}")
        print(f"  Scheduled count: {info.get('scheduled_count', 0)}")
    
    # Load and test trained model
    print("\n" + "-"*40)
    print("Testing trained model (500k):")
    
    model_path = "models/multidiscrete/simple/model_500k.zip"
    if Path(model_path).exists():
        model = PPO.load(model_path)
        vec_env = DummyVecEnv([lambda: env])
        
        obs = vec_env.reset()
        
        # Get model predictions
        print("\nModel action predictions (first 10 steps):")
        for step in range(10):
            action, _ = model.predict(obs, deterministic=True)
            action = action[0]  # Unwrap from batch
            
            print(f"  Step {step}: Job {action[0]}, Machine {action[1]}")
            
            # Check validity
            if hasattr(env, '_compatibility_matrix'):
                is_valid = env._compatibility_matrix[action[0], action[1]]
                print(f"    Valid: {is_valid}")
            
            obs, reward, done, info = vec_env.step([action])
            if info[0].get('invalid_action', False):
                print(f"    Result: Invalid action!")
            else:
                print(f"    Result: Scheduled! Count: {info[0].get('scheduled_count', 0)}")
    
    # Analyze action distribution
    print("\n" + "-"*40)
    print("Analyzing job-machine compatibility:")
    
    if hasattr(env, '_compatibility_matrix'):
        compat_matrix = env._compatibility_matrix
        
        # Jobs with no compatible machines
        no_compat_jobs = []
        for i in range(len(env.jobs)):
            if np.sum(compat_matrix[i]) == 0:
                no_compat_jobs.append(i)
        
        print(f"Jobs with NO compatible machines: {len(no_compat_jobs)}")
        if no_compat_jobs:
            print(f"  Job indices: {no_compat_jobs[:10]}...")  # First 10
        
        # Machines that can't process any jobs
        no_compat_machines = []
        for j in range(env.n_machines):
            if np.sum(compat_matrix[:, j]) == 0:
                no_compat_machines.append(j)
        
        print(f"Machines that can't process ANY jobs: {len(no_compat_machines)}")
        if no_compat_machines:
            print(f"  Machine indices: {no_compat_machines[:10]}...")

if __name__ == "__main__":
    debug_action_space()