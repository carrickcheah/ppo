#!/usr/bin/env python3
"""
Check job count mismatch issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def check_mismatch():
    print("\nChecking job count mismatch...\n")
    
    # Create environment requesting 411 jobs
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=100,
        seed=42
    )
    
    print(f"1. Environment creation:")
    print(f"   Requested n_jobs: 411")
    print(f"   Action space: {env.action_space}")
    print(f"   Action space dimensions: {env.action_space.nvec}")
    
    # Reset to load data
    obs, info = env.reset()
    
    print(f"\n2. After reset:")
    print(f"   Actual jobs loaded: {len(env.jobs) if hasattr(env, 'jobs') and env.jobs else 'Unknown'}")
    print(f"   Actual machines: {len(env.machines) if hasattr(env, 'machines') and env.machines else 'Unknown'}")
    
    # Check internal n_jobs
    if hasattr(env, 'n_jobs'):
        print(f"   Internal n_jobs: {env.n_jobs}")
    
    # Check job mask
    if hasattr(env, 'job_mask'):
        print(f"   Job mask length: {len(env.job_mask)}")
        print(f"   Valid jobs in mask: {sum(env.job_mask)}")
    
    # Try different n_jobs values
    print(f"\n3. Testing with different n_jobs:")
    
    for n_jobs in [100, 172, 200, 411]:
        env2 = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=n_jobs,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=100,
            seed=42
        )
        env2.reset()
        actual_jobs = len(env2.jobs) if hasattr(env2, 'jobs') and env2.jobs else 0
        print(f"   n_jobs={n_jobs} -> actual jobs: {actual_jobs}, action space: {env2.action_space.nvec}")

if __name__ == "__main__":
    check_mismatch()