#!/usr/bin/env python3
"""
Debug why all actions are invalid in MultiDiscrete environment
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def debug_invalid_actions():
    """Debug the invalid action issue."""
    print("\n" + "="*60)
    print("Debugging Invalid Actions in MultiDiscrete Environment")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=100,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=42
    )
    
    obs, info = env.reset()
    
    print("1. Environment Info:")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Jobs available: {len(env.jobs)}")
    print(f"   - Machines available: {len(env.machines)}")
    print(f"   - Compatibility matrix shape: {env.compatibility_matrix.shape if env.compatibility_matrix is not None else 'None'}")
    
    print("\n2. Action Masks:")
    masks = info.get('action_masks', {})
    job_mask = masks.get('job', np.array([]))
    machine_masks = masks.get('machine', np.array([]))
    
    print(f"   - Job mask shape: {job_mask.shape}")
    print(f"   - Job mask sum: {np.sum(job_mask)} (valid jobs)")
    print(f"   - Machine masks shape: {machine_masks.shape}")
    
    print("\n3. Checking Job-Machine Compatibility:")
    if env.compatibility_matrix is not None:
        print(f"   - Compatibility matrix shape: {env.compatibility_matrix.shape}")
        print(f"   - Total compatible pairs: {np.sum(env.compatibility_matrix)}")
        print(f"   - Average compatible machines per job: {np.mean(np.sum(env.compatibility_matrix, axis=1)):.1f}")
        
        # Check first few jobs
        for i in range(min(5, len(env.jobs))):
            compatible_machines = np.sum(env.compatibility_matrix[i])
            print(f"   - Job {i}: {compatible_machines} compatible machines")
    
    print("\n4. Testing Specific Actions:")
    
    # Test action [0, 0]
    print("\n   Testing action [0, 0]:")
    action = np.array([0, 0])
    obs, reward, done, truncated, info = env.step(action)
    print(f"   - Invalid: {info.get('invalid_action', False)}")
    print(f"   - Reason: {info.get('invalid_reason', 'N/A')}")
    
    # Check job mask attribute
    print(f"\n5. Checking job_mask attribute:")
    print(f"   - hasattr(env, 'job_mask'): {hasattr(env, 'job_mask')}")
    if hasattr(env, 'job_mask'):
        print(f"   - env.job_mask shape: {env.job_mask.shape}")
        print(f"   - env.job_mask[0]: {env.job_mask[0]}")
    
    # Check if the issue is in parent class
    print(f"\n6. Parent class attributes:")
    print(f"   - Jobs length: {len(env.jobs) if hasattr(env, 'jobs') else 'No jobs attr'}")
    print(f"   - Scheduled count: {env.scheduled_count if hasattr(env, 'scheduled_count') else 'No scheduled_count'}")
    
    # Test finding a valid action manually
    print("\n7. Finding valid action manually:")
    env.reset()  # Reset to clear state
    
    # Initialize job_mask if not present
    if not hasattr(env, 'job_mask'):
        print("   - Creating job_mask manually")
        env.job_mask = np.ones(len(env.jobs), dtype=bool)
    
    # Find first compatible pair
    if env.compatibility_matrix is not None:
        for job_idx in range(min(10, len(env.jobs))):
            for machine_idx in range(len(env.machines)):
                if env.compatibility_matrix[job_idx, machine_idx]:
                    print(f"   - Found compatible: Job {job_idx} -> Machine {machine_idx}")
                    
                    # Test this action
                    action = np.array([job_idx, machine_idx])
                    obs, reward, done, truncated, info = env.step(action)
                    print(f"   - Result: Invalid={info.get('invalid_action', False)}, Reward={reward:.2f}")
                    if not info.get('invalid_action', False):
                        print("   ✅ Successfully scheduled a job!")
                    else:
                        print(f"   - Failure reason: {info.get('invalid_reason', 'Unknown')}")
                    break
            if not info.get('invalid_action', False):
                break

if __name__ == "__main__":
    debug_invalid_actions()