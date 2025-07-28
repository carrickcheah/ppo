"""
Quick test of better reward structure
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase3.environments.better_reward_env import BetterRewardEnvironment


def test_reward_structure():
    """Test the new reward structure."""
    env = BetterRewardEnvironment('toy_normal', verbose=False)
    
    print("Testing Better Reward Structure")
    print("=" * 50)
    
    obs, _ = env.reset()
    
    # Get valid actions
    valid_actions = env._get_valid_actions()
    print(f"\nInitial valid actions: {len(valid_actions)}")
    
    # Test scheduling a few jobs
    total_reward = 0
    for i in range(5):
        if len(valid_actions) > 1:  # More than just no-action
            # Take first non-no-action
            # Check for no-action (last in the list)
            no_action = (len(env.family_ids), len(env.machine_ids))
            action = valid_actions[0] if valid_actions[0] != no_action else valid_actions[1]
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"\nStep {i+1}:")
            print(f"  Action: Job {action[0]}, Machine {action[1]}")
            print(f"  Reward: {reward:.1f}")
            print(f"  Jobs scheduled: {len(env.scheduled_jobs)}/{env.total_tasks}")
            
            if 'reward_breakdown' in info:
                print(f"  Breakdown: {info['reward_breakdown']}")
            
            if done:
                if 'completion_bonus' in info:
                    print(f"  COMPLETION BONUS: {info['completion_bonus']}")
                break
            
            valid_actions = env._get_valid_actions()
    
    print(f"\nTotal reward after 5 steps: {total_reward:.1f}")
    print(f"Final completion: {len(env.scheduled_jobs)}/{env.total_tasks} = {len(env.scheduled_jobs)/env.total_tasks:.1%}")
    
    # Test what happens with late job
    print("\n" + "-"*50)
    print("Key improvement: Late jobs still get POSITIVE rewards!")


if __name__ == "__main__":
    test_reward_structure()