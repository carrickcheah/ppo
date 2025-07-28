"""
Quick test of schedule all environment
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase3.environments.schedule_all_env import ScheduleAllEnvironment


def test_schedule_all():
    """Test the environment."""
    env = ScheduleAllEnvironment('toy_normal', verbose=True)
    
    print("\nTesting Schedule All Environment")
    print("=" * 50)
    
    obs, _ = env.reset()
    
    # Try to schedule all jobs
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 50:
        valid_actions = env._get_valid_actions()
        
        if len(valid_actions) > 1:
            # Take first non-no-action
            no_action = (len(env.family_ids), len(env.machine_ids))
            action = valid_actions[0] if valid_actions[0] != no_action else valid_actions[1]
        else:
            action = valid_actions[0]
        
        obs, reward, done, truncated, info = env.step(np.array(action))
        total_reward += reward
        steps += 1
        
        print(f"\nStep {steps}: Reward={reward:.1f}, Scheduled={len(env.scheduled_jobs)}/{env.total_tasks}")
        
        if done:
            print(f"\nEpisode ended!")
            if 'all_tasks_scheduled' in info:
                print("✓ All tasks scheduled!")
            elif 'no_valid_actions' in info:
                print("✗ No more valid actions")
            break
    
    print(f"\nFinal: {len(env.scheduled_jobs)}/{env.total_tasks} scheduled")
    print(f"Total reward: {total_reward:.1f}")


if __name__ == "__main__":
    test_schedule_all()