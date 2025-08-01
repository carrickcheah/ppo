"""
Test the bottleneck environment to diagnose issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environments.small_bottleneck_env import SmallBottleneckEnvironment

def test_environment():
    """Test basic environment functionality."""
    env = SmallBottleneckEnvironment(verbose=True)
    
    print("\n=== Environment Info ===")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Number of families: {len(env.families)}")
    print(f"Number of machines: {len(env.machines)}")
    print(f"Total tasks: {env.total_tasks}")
    print(f"Max steps: {env.max_steps}")
    
    # Reset and check initial state
    obs, info = env.reset()
    print(f"\n=== Initial State ===")
    print(f"Observation shape: {obs.shape}")
    print(f"Valid actions available: {len(info['valid_actions'])}")
    print(f"Initial time: {env.current_time}")
    
    # Test a few random actions
    print(f"\n=== Testing Random Actions ===")
    total_reward = 0
    valid_schedules = 0
    invalid_actions = 0
    no_actions = 0
    
    for i in range(50):
        # Get random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Current time: {env.current_time:.2f}")
        print(f"  Scheduled jobs: {len(env.scheduled_jobs)}")
        print(f"  Action type: {info.get('action_type', 'unknown')}")
        
        if info.get('action_type') == 'schedule':
            valid_schedules += 1
            print(f"  Scheduled: {info.get('scheduled_job')}")
        elif info.get('action_type') == 'invalid':
            invalid_actions += 1
        elif info.get('action_type') == 'no_action':
            no_actions += 1
            
        if done or truncated:
            print(f"\nEpisode ended!")
            break
    
    print(f"\n=== Summary ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Valid schedules: {valid_schedules}")
    print(f"Invalid actions: {invalid_actions}")
    print(f"No actions: {no_actions}")
    print(f"Final time: {env.current_time:.2f}")
    print(f"Completion rate: {len(env.scheduled_jobs) / env.total_tasks * 100:.1f}%")
    
    # Test valid actions
    print(f"\n=== Testing Valid Actions ===")
    env.reset()
    valid_actions = env._get_valid_actions()
    print(f"Number of valid actions: {len(valid_actions)}")
    
    if valid_actions:
        # Try first valid action
        action = valid_actions[0]
        obs, reward, done, truncated, info = env.step(action)
        print(f"First valid action: {action}")
        print(f"Reward: {reward:.2f}")
        print(f"Action type: {info.get('action_type')}")
        
    # Check action space bounds
    print(f"\n=== Action Space Analysis ===")
    print(f"Family indices: 0 to {len(env.family_ids)-1}")
    print(f"Machine indices: 0 to {len(env.machine_ids)-1}")
    print(f"NO-ACTION family index: {len(env.family_ids)}")
    print(f"NO-ACTION machine index: {len(env.machine_ids)}")
    
    # Check time progression
    print(f"\n=== Time Progression Test ===")
    env.reset()
    times = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        times.append(env.current_time)
    print(f"Time after 10 steps: {times}")
    print(f"Time increment per step: {times[1] - times[0] if len(times) > 1 else 'N/A'}")


if __name__ == "__main__":
    test_environment()