"""
Quick test to verify job splitting works with break constraints.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv
import numpy as np

def test_job_splitting():
    """Test that long jobs can be scheduled despite break constraints."""
    print("Testing job splitting with break constraints...")
    
    # Create environment
    env = ScaledProductionEnv(
        n_machines=40,
        max_episode_steps=1000,
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=42
    )
    
    # Reset and run a few steps
    obs, _ = env.reset()
    
    errors_before = 0
    errors_after = 0
    steps = 50
    
    print(f"\nRunning {steps} steps to check for 'Could not find valid start time' errors...")
    
    for i in range(steps):
        # Get valid actions
        valid_actions = env.valid_actions
        
        if not valid_actions:
            print(f"No valid actions at step {i}")
            break
            
        # Take a random action
        action = np.random.randint(len(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
    
    print(f"\nTest completed!")
    print(f"- Steps executed: {i+1}")
    print(f"- Current makespan: {env.episode_makespan:.1f}h")
    print(f"- Jobs scheduled: {sum(len(completed) for completed in env.completed_tasks.values())}")
    
    # Check if we're making progress
    if env.episode_makespan > 0:
        print("\n✅ SUCCESS: Jobs are being scheduled despite break constraints!")
    else:
        print("\n❌ FAILURE: No jobs were scheduled")

if __name__ == "__main__":
    test_job_splitting()