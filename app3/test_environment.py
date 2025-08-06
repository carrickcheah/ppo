"""
Test script to verify the scheduling environment works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from environments.scheduling_env import SchedulingEnv


def test_environment():
    """Test basic environment functionality."""
    print("Testing Scheduling Environment")
    print("=" * 50)
    
    # Create environment
    env = SchedulingEnv("data/10_jobs.json", max_steps=100)
    print(f"✓ Environment created")
    print(f"  - Tasks: {env.n_tasks}")
    print(f"  - Machines: {env.n_machines}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space.shape}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Environment reset")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Valid actions: {info['valid_actions']}")
    
    # Run episode
    print(f"\n Running episode...")
    print("-" * 30)
    
    total_reward = 0
    steps = 0
    
    while steps < 50:  # Limit steps for testing
        # Get valid actions
        valid_actions = np.where(info['action_mask'])[0]
        
        if len(valid_actions) == 0:
            print(f"Step {steps}: No valid actions available")
            break
            
        # Choose random valid action
        action = np.random.choice(valid_actions)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print progress every 10 steps
        if steps % 10 == 0:
            print(f"Step {steps}: Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}, "
                  f"Reward: {reward:.2f}, Utilization: {info['utilization']:.2%}")
        
        if terminated or truncated:
            break
    
    # Final results
    print("-" * 30)
    print(f"\n✓ Episode completed")
    print(f"  - Steps taken: {steps}")
    print(f"  - Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Final utilization: {info['utilization']:.2%}")
    
    # Get final schedule
    schedule = env.get_final_schedule()
    metrics = schedule['metrics']
    
    print(f"\n✓ Final metrics:")
    print(f"  - On-time rate: {metrics.get('on_time_rate', 0):.2%}")
    print(f"  - Early rate: {metrics.get('early_rate', 0):.2%}")
    print(f"  - Late rate: {metrics.get('late_rate', 0):.2%}")
    
    # Show some scheduled tasks
    if schedule['tasks']:
        print(f"\n✓ Sample scheduled tasks:")
        for task in schedule['tasks'][:5]:
            print(f"  - {task['task_id']} on {task['machine']}: "
                  f"{task['start']:.1f}-{task['end']:.1f}h "
                  f"(LCD: {task['lcd_days']} days)")
    
    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    test_environment()