#!/usr/bin/env python3
"""
Debug why trained model produces invalid actions
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
    print("Debugging Action Space Issues")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=100,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=200,
        invalid_action_penalty=-5.0,
        seed=42
    )
    
    print(f"1. Environment Info:")
    print(f"   Action space: {env.action_space}")
    print(f"   Action space type: {type(env.action_space)}")
    print(f"   Action shape: {env.action_space.nvec}")
    print(f"   Observation space: {env.observation_space}")
    
    # Reset and check masks
    obs, info = env.reset()
    masks = info.get('action_masks', {})
    job_mask = masks.get('job', np.array([]))
    
    print(f"\n2. Initial State:")
    print(f"   Valid jobs: {np.sum(job_mask)}/{len(job_mask)}")
    print(f"   First 10 job mask: {job_mask[:10]}")
    
    # Try a valid action manually
    print(f"\n3. Testing Manual Actions:")
    
    # Find first valid job
    valid_job_idx = None
    for i in range(len(job_mask)):
        if job_mask[i]:
            valid_job_idx = i
            break
    
    if valid_job_idx is not None:
        # Find valid machine for this job
        machine_masks = masks.get('machine', np.array([]))
        valid_machine_idx = None
        
        if valid_job_idx < len(machine_masks):
            for j in range(len(machine_masks[valid_job_idx])):
                if machine_masks[valid_job_idx][j]:
                    valid_machine_idx = j
                    break
        
        if valid_machine_idx is not None:
            action = np.array([valid_job_idx, valid_machine_idx])
            print(f"   Valid action found: job={valid_job_idx}, machine={valid_machine_idx}")
            
            obs, reward, done, truncated, info = env.step(action)
            print(f"   Result: reward={reward:.2f}, invalid={info.get('invalid_action', False)}")
    
    # Load trained model
    print(f"\n4. Testing Trained Model:")
    vec_env = DummyVecEnv([lambda: env])
    
    try:
        model = PPO.load("models/multidiscrete/simple_model")
        print("   Model loaded successfully")
        
        # Get raw action from model
        obs = vec_env.reset()
        
        # Debug model output
        with model.policy.predict_deterministic(obs) as prediction:
            raw_action = prediction
        print(f"   Raw model output: {raw_action}")
        print(f"   Raw action shape: {raw_action.shape}")
        
        # Get proper prediction
        action, _ = model.predict(obs, deterministic=True)
        print(f"   Predicted action: {action}")
        print(f"   Action type: {type(action)}")
        print(f"   Action shape: {action.shape if hasattr(action, 'shape') else 'no shape'}")
        
    except Exception as e:
        print(f"   Error loading model: {e}")
    
    # Check if action space mismatch
    print(f"\n5. Action Space Debugging:")
    sample = env.action_space.sample()
    print(f"   Sample action: {sample}")
    print(f"   Sample type: {type(sample)}")
    print(f"   Sample shape: {sample.shape}")
    
    # Test DummyVecEnv wrapping
    print(f"\n6. VecEnv Action Space:")
    vec_env = DummyVecEnv([lambda: env])
    print(f"   VecEnv action space: {vec_env.action_space}")
    vec_sample = vec_env.action_space.sample()
    print(f"   VecEnv sample: {vec_sample}")
    print(f"   VecEnv sample type: {type(vec_sample)}")

if __name__ == "__main__":
    debug_action_space()