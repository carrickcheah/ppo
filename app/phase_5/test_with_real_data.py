#!/usr/bin/env python3
"""
Test Phase 5 model with correct real data dimensions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def test_with_real_data():
    print("\n" + "="*60)
    print("Testing Phase 5 with Real Data (320 jobs)")
    print("="*60 + "\n")
    
    # Create environment with correct dimensions
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,  # Match real data
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=42
    )
    
    # Reset and check
    obs, info = env.reset()
    print(f"Environment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Jobs loaded: {len(env.jobs)}")
    print(f"  Machines loaded: {len(env.machines)}")
    
    # Test random actions
    print("\n" + "-"*40)
    print("Testing random actions:")
    valid_count = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if not info.get('invalid_action', False):
            valid_count += 1
            print(f"  Valid: Job {action[0]} → Machine {action[1]}")
    
    print(f"\nRandom valid rate: {valid_count}/20 ({valid_count/20*100:.1f}%)")
    
    # Test trained model
    print("\n" + "-"*40)
    print("Testing trained model:")
    
    model_path = "models/multidiscrete/simple/model_500k.zip"
    if Path(model_path).exists():
        # Load model
        model = PPO.load(model_path)
        
        # The model was trained with wrong dimensions, so let's check
        print(f"Model policy expects input shape: {model.policy.observation_space.shape}")
        print(f"Model policy expects action space: {model.policy.action_space}")
        
        # Create new environment matching model's expectations
        env_for_model = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=411,  # What model was trained with
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=500,
            seed=42
        )
        vec_env = DummyVecEnv([lambda: env_for_model])
        
        obs = vec_env.reset()
        
        print("\nModel predictions (10 steps):")
        scheduled = 0
        invalid = 0
        
        for step in range(10):
            action, _ = model.predict(obs, deterministic=True)
            action = action[0]
            
            print(f"  Step {step}: Job {action[0]}, Machine {action[1]}")
            
            # Check if job index is within real data bounds
            if action[0] >= 320:
                print(f"    ERROR: Job index {action[0]} exceeds real data (320 jobs)")
                invalid += 1
            else:
                obs, reward, done, info = vec_env.step([action])
                if info[0].get('invalid_action', False):
                    print(f"    Invalid: {info[0].get('invalid_reason', 'Unknown')}")
                    invalid += 1
                else:
                    scheduled = info[0].get('scheduled_count', 0)
                    print(f"    Valid! Scheduled count: {scheduled}")
        
        print(f"\nModel results: {scheduled} scheduled, {invalid} invalid")
    
    # Test with correct dimensions
    print("\n" + "-"*40)
    print("Creating fresh model with correct dimensions:")
    
    # Create environment with real data dimensions
    env_correct = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,  # Real data
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=42
    )
    vec_env_correct = DummyVecEnv([lambda: env_correct])
    
    print(f"New environment action space: {vec_env_correct.action_space}")
    print("\nThis is the correct setup for training!")

if __name__ == "__main__":
    test_with_real_data()