#!/usr/bin/env python3
"""
Test if the environment properly handles actions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.full_production_env import FullProductionEnv
import numpy as np

print("TESTING ACTION HANDLING")
print("="*40)

# Create environment
env = FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    max_valid_actions=888,
    max_episode_steps=3000,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)

obs, info = env.reset()
print(f"Initial state:")
print(f"- Valid actions available: {len(env.valid_actions)}")
print(f"- Action space: {env.action_space}")
print(f"- First few valid actions: {env.valid_actions[:5] if env.valid_actions else 'None'}")

# Test different action values
print(f"\nTesting actions:")

# Test 1: Valid action (within valid_actions range)
if env.valid_actions:
    action = 0  # First valid action
    print(f"\n1. Action {action} (should be valid)")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Result: reward={reward:.2f}, done={terminated or truncated}")
    
    # Test 2: Out of range but within action space
    action = 100
    print(f"\n2. Action {action} (likely invalid)")
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Result: reward={reward:.2f}, done={terminated or truncated}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Way out of range
    action = 500
    print(f"\n3. Action {action} (definitely invalid)")
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Result: reward={reward:.2f}, done={terminated or truncated}")
    except Exception as e:
        print(f"   Error: {e}")

print("\n" + "="*40)
print("CONCLUSION:")
print("The action value is used as an INDEX into valid_actions list.")
print("If there are 50 valid actions, only actions 0-49 are valid.")
print(f"With max_valid_actions={env.max_valid_actions}, the action space is 0-{env.action_space.n-1}")
print("But the actual valid range depends on current valid_actions!")

env.close()