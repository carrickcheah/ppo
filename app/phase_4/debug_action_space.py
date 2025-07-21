#!/usr/bin/env python3
"""
Debug the action space to understand why model fails with 1000 actions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.full_production_env import FullProductionEnv
import numpy as np

print("DEBUGGING ACTION SPACE ISSUES")
print("="*40)

# Test different action space sizes
for max_actions in [200, 400, 600, 1000]:
    print(f"\nTesting with max_valid_actions={max_actions}")
    
    env = FullProductionEnv(
        n_machines=150,
        n_jobs=500,
        max_valid_actions=max_actions,
        max_episode_steps=3000,
        state_compression="hierarchical",
        use_break_constraints=True,
        use_holiday_constraints=True,
        seed=42
    )
    
    obs, info = env.reset()
    print(f"- Initial valid actions: {len(env.valid_actions)}")
    print(f"- Action space size: {env.action_space.n}")
    
    # Take some random valid actions
    scheduled = 0
    for step in range(10):
        if not env.valid_actions:
            print(f"  No valid actions at step {step}")
            break
            
        # Pick a VALID action (not random from full space)
        valid_action_idx = np.random.randint(0, len(env.valid_actions))
        actual_action = valid_action_idx  # This might be the issue!
        
        obs, reward, done, info = env.step(actual_action)
        
        if hasattr(env, 'completed_tasks'):
            new_scheduled = sum(len(tasks) for tasks in env.completed_tasks.values())
            if new_scheduled > scheduled:
                scheduled = new_scheduled
                print(f"  Step {step}: Scheduled job {scheduled}, {len(env.valid_actions)} actions remain")
    
    env.close()

print("\n" + "="*40)
print("ANALYSIS:")
print("1. Action space size doesn't match valid actions")
print("2. The model outputs actions in range [0, action_space.n)")
print("3. But valid actions might be indexed differently")
print("\nThe issue is likely in how actions are mapped!")