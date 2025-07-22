#!/usr/bin/env python3
"""
Debug environment attributes to understand invalid actions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def debug_environment():
    print("\n" + "="*60)
    print("Debugging Environment Attributes")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=100,
        seed=42
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Check all attributes
    print("Environment attributes:")
    attrs = dir(env)
    
    # Look for compatibility-related attributes
    compat_attrs = [attr for attr in attrs if 'compat' in attr.lower() or 'matrix' in attr.lower()]
    print(f"\nCompatibility-related attributes: {compat_attrs}")
    
    # Check specific attributes
    print("\nChecking key attributes:")
    print(f"  has 'compatibility_matrix': {hasattr(env, 'compatibility_matrix')}")
    print(f"  has '_compatibility_matrix': {hasattr(env, '_compatibility_matrix')}")
    print(f"  has 'job_mask': {hasattr(env, 'job_mask')}")
    print(f"  has 'jobs': {hasattr(env, 'jobs')}")
    print(f"  has 'machines': {hasattr(env, 'machines')}")
    print(f"  has 'n_jobs': {hasattr(env, 'n_jobs')} = {getattr(env, 'n_jobs', None)}")
    print(f"  has 'n_machines': {hasattr(env, 'n_machines')} = {getattr(env, 'n_machines', None)}")
    
    if hasattr(env, 'compatibility_matrix'):
        cm = env.compatibility_matrix
        if cm is not None:
            print(f"\nCompatibility matrix shape: {cm.shape}")
            print(f"Type: {type(cm)}")
            print(f"Any True values: {np.any(cm) if isinstance(cm, np.ndarray) else 'N/A'}")
        else:
            print("\nCompatibility matrix is None!")
    
    # Test action validity directly
    print("\n" + "-"*40)
    print("Testing action validity check:")
    
    # Test a specific action
    test_actions = [
        (0, 0),    # First job, first machine
        (10, 10),  # Middle
        (100, 50), # Later job
        (333, 118) # The one model keeps selecting
    ]
    
    for job_idx, machine_idx in test_actions:
        print(f"\nTesting Job {job_idx} → Machine {machine_idx}:")
        
        # Call the validity check if it exists
        if hasattr(env, '_check_action_validity'):
            is_valid, reason = env._check_action_validity(job_idx, machine_idx)
            print(f"  Valid: {is_valid}")
            if not is_valid:
                print(f"  Reason: {reason}")
        
        # Try the action
        action = np.array([job_idx, machine_idx])
        try:
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step result: reward={reward:.2f}, invalid={info.get('invalid_action', False)}")
            if info.get('invalid_action'):
                print(f"  Invalid reason: {info.get('invalid_reason', 'Unknown')}")
        except Exception as e:
            print(f"  Step failed: {e}")
    
    # Check action masks
    print("\n" + "-"*40)
    print("Checking action masks:")
    
    if hasattr(env, 'get_action_masks'):
        masks = env.get_action_masks()
        print(f"Action masks keys: {list(masks.keys())}")
        
        if 'job' in masks:
            job_mask = masks['job']
            print(f"  Job mask shape: {job_mask.shape}")
            print(f"  Available jobs: {np.sum(job_mask)}")
            print(f"  First 10 job mask: {job_mask[:10].tolist()}")
        
        if 'machine' in masks:
            machine_masks = masks['machine']
            if isinstance(machine_masks, dict):
                print(f"  Machine masks: dict with {len(machine_masks)} entries")
                # Check first job's machine mask
                if 0 in machine_masks:
                    print(f"  Job 0 machine mask shape: {machine_masks[0].shape}")
                    print(f"  Job 0 compatible machines: {np.sum(machine_masks[0])}")
            elif isinstance(machine_masks, np.ndarray):
                print(f"  Machine masks shape: {machine_masks.shape}")

if __name__ == "__main__":
    debug_environment()