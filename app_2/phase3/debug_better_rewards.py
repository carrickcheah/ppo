"""
Debug the better reward environment to see what's happening
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase3.environments.better_reward_env import BetterRewardEnvironment


def debug_rewards():
    """Debug reward calculation."""
    env = BetterRewardEnvironment('toy_normal', verbose=False)
    
    print("Reward Config:")
    for key, value in env.reward_config.items():
        print(f"  {key}: {value}")
    
    print("\nTesting a few steps:")
    
    obs, _ = env.reset()
    total_reward = 0
    
    # Take no-action
    no_action = np.array([len(env.family_ids), len(env.machine_ids)])
    obs, reward, done, truncated, info = env.step(no_action)
    total_reward += reward
    print(f"\nNo-action reward: {reward}")
    
    # Take an invalid action
    invalid_action = np.array([99, 99])
    obs, reward, done, truncated, info = env.step(invalid_action)
    total_reward += reward
    print(f"Invalid action reward: {reward}")
    
    # Schedule a valid job
    valid_actions = env._get_valid_actions()
    if len(valid_actions) > 1:
        action = valid_actions[0]
        obs, reward, done, truncated, info = env.step(np.array(action))
        total_reward += reward
        print(f"Valid schedule reward: {reward}")
        print(f"Info: {info}")


if __name__ == "__main__":
    debug_rewards()