"""
Debug what the simple fix model is doing
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


def debug_model_behavior():
    """Debug what the model is actually doing."""
    
    # Load latest checkpoint
    model_path = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/simple_fix/toy_normal/checkpoint_700000_steps.zip"
    model = PPO.load(model_path)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed('toy_normal', verbose=False)
    
    print("DEBUGGING MODEL BEHAVIOR")
    print("=" * 50)
    print(f"Late penalty: {env.reward_config['late_penalty_per_day']} per day")
    
    obs, _ = env.reset()
    
    # Take 20 steps to see pattern
    total_reward = 0
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        action_type = info.get('action_type', 'unknown')
        
        if action_type == 'no_action':
            print(f"Step {step+1}: NO-ACTION (reward: {reward})")
        elif action_type == 'schedule':
            job = info['scheduled_job']
            print(f"Step {step+1}: Scheduled {job} (reward: {reward})")
        else:
            print(f"Step {step+1}: {action_type} action {action} (reward: {reward})")
        
        if done or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break
    
    print(f"\nTotal reward: {total_reward}")
    print(f"Jobs scheduled: {len(env.scheduled_jobs)}/{env.total_tasks}")
    
    # Check what valid actions remain
    valid_actions = env._get_valid_actions()
    print(f"\nValid actions remaining: {len(valid_actions)}")
    
    # Check rewards for scheduling the impossible job
    print("\n" + "-"*50)
    print("Checking reward structure:")
    
    # Find JOAW25050075 (the impossible job)
    for fid, family in env.families.items():
        if 'JOAW25050075' in fid:
            total_time = sum(task['processing_time'] for task in family['tasks'])
            deadline = family['lcd_days_remaining'] * 24
            late_hours = total_time - deadline
            late_days = late_hours / 24
            late_penalty = env.reward_config['late_penalty_per_day'] * late_days
            
            print(f"\nImpossible job {fid}:")
            print(f"  Processing time: {total_time:.1f} hours")
            print(f"  Deadline: {deadline:.1f} hours")
            print(f"  Will be late by: {late_hours:.1f} hours ({late_days:.1f} days)")
            print(f"  Late penalty: {late_penalty:.1f}")
            print(f"  Total reward if scheduled: {10 + 20 + late_penalty:.1f}")


if __name__ == "__main__":
    debug_model_behavior()