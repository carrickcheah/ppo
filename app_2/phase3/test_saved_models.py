"""
Test the models that were just trained
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed
import numpy as np

print("Testing Saved Models")
print("="*60)

# Check for saved models
model_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models"
stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']

for stage in stages:
    model_path = os.path.join(model_dir, f"{stage}_final.zip")
    
    if not os.path.exists(model_path):
        print(f"\n{stage}: Model not found")
        continue
    
    print(f"\n{stage}:")
    print("-"*40)
    
    # Load model and create env
    model = PPO.load(model_path)
    env = CurriculumEnvironmentTrulyFixed(stage, verbose=False)
    
    # Test on 10 episodes
    total_scheduled = 0
    total_possible = 0
    rewards = []
    
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            steps += 1
        
        scheduled = len(env.scheduled_jobs)
        total = env.total_tasks
        total_scheduled += scheduled
        total_possible += total
        rewards.append(ep_reward)
        
        if ep == 0:  # Show first episode details
            print(f"  Episode 1: {scheduled}/{total} jobs ({scheduled/total*100:.0f}%)")
            print(f"  Action breakdown: {info}")
    
    # Overall stats
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    print(f"  Average scheduling rate: {final_rate:.1%}")
    print(f"  Average reward: {np.mean(rewards):.1f}")
    
    if final_rate >= 0.8:
        print(f"  Status: EXCELLENT! Target achieved!")
    elif final_rate >= 0.5:
        print(f"  Status: Good progress")
    else:
        print(f"  Status: Needs more training")

print("\n" + "="*60)