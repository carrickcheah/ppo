"""Debug script to understand the curriculum environment."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv
import numpy as np


def debug_environment():
    """Debug the environment to see what's happening."""
    print("DEBUGGING CURRICULUM ENVIRONMENT")
    print("="*60)
    
    # Create environment without breaks
    env = ScaledProductionEnv(
        n_machines=40,
        use_break_constraints=False,
        seed=42
    )
    
    # Reset
    obs, info = env.reset()
    
    print(f"Environment created successfully")
    print(f"Number of jobs: {env.n_jobs}")
    print(f"Number of families: {env.n_families}")
    print(f"Number of machines: {env.n_machines}")
    print(f"Action space: {env.action_space}")
    print(f"Max valid actions: {env.max_valid_actions}")
    
    # Check valid actions
    print(f"\nValid actions available: {len(env.valid_actions)}")
    if env.valid_actions:
        print("First 5 valid actions:")
        for i, (family_id, task_idx, task) in enumerate(env.valid_actions[:5]):
            print(f"  {i}: Family {family_id}, Task {task_idx}, Sequence {task.get('sequence', 'N/A')}")
    
    # Try to take an action
    if env.valid_actions:
        print("\nTaking first valid action...")
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Info keys: {list(info.keys())}")
        for key, value in info.items():
            if key != 'TimeLimit.truncated':
                print(f"  {key}: {value}")
        
        # Check machine loads
        print(f"\nMachine loads after action:")
        non_zero_loads = [(i, load) for i, load in enumerate(env.machine_loads) if load > 0]
        for machine_id, load in non_zero_loads[:5]:
            print(f"  Machine {machine_id}: {load:.1f}h")
    
    print("\nDiagnosis complete!")


if __name__ == "__main__":
    debug_environment()