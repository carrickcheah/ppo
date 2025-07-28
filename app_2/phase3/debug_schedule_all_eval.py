"""
Debug why evaluation isn't working properly
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from phase3.environments.schedule_all_env import ScheduleAllEnvironment


def debug_evaluation():
    """Debug the evaluation issue."""
    
    # Load model
    model_path = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_schedule_all/toy_normal_schedule_all.zip"
    model = PPO.load(model_path)
    
    # Create environment with verbose
    env = ScheduleAllEnvironment('toy_normal', verbose=True)
    
    print("\nDEBUGGING EVALUATION")
    print("=" * 50)
    
    obs, _ = env.reset()
    done = False
    steps = 0
    
    print(f"\nInitial state:")
    print(f"Total tasks to schedule: {env.total_tasks}")
    print(f"Max steps: {env.max_steps}")
    
    # Take a few steps
    for i in range(10):
        if done:
            break
            
        action, _ = model.predict(obs, deterministic=True)
        print(f"\nStep {i+1}: Action = {action}")
        
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        print(f"Reward: {reward:.1f}")
        print(f"Done: {done}, Truncated: {truncated}")
        print(f"Scheduled: {len(env.scheduled_jobs)}/{env.total_tasks}")
        print(f"Info: {info}")
        
        if done:
            print("\nEpisode ended!")
            break
    
    # Check what's happening
    print(f"\nCurrent timestep: {env.steps}")
    print(f"Max steps: {env.max_steps}")
    
    # Check valid actions
    valid_actions = env._get_valid_actions()
    print(f"\nValid actions remaining: {len(valid_actions)}")
    no_action = (len(env.family_ids), len(env.machine_ids))
    print(f"No-action: {no_action}")
    
    # Check if model is just choosing no-action
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
        action_tuple = tuple(action)
        if action_tuple == no_action:
            print(f"Model chose NO-ACTION")
        else:
            print(f"Model chose: {action_tuple}")


if __name__ == "__main__":
    debug_evaluation()