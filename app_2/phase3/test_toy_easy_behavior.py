"""Test toy_easy behavior to diagnose 0% scheduling rate"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal
from stable_baselines3 import PPO
import numpy as np

# Create environment
env = CurriculumEnvironmentReal(stage_name='toy_easy', verbose=True)

# Check initial state
obs, info = env.reset()
print(f"\nInitial observation shape: {obs.shape}")
print(f"Valid actions available: {len(info.get('valid_actions', []))}")
print(f"Sample valid actions: {info.get('valid_actions', [])[:5]}")

# Try loading the trained model
model_path = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/toy_easy/final_model.zip"
if os.path.exists(model_path):
    print(f"\nLoading trained model from {model_path}")
    model = PPO.load(model_path)
    
    # Run one episode
    obs, _ = env.reset()
    done = False
    step = 0
    valid_actions = 0
    invalid_actions = 0
    
    while not done and step < 100:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get('action_valid', False):
            valid_actions += 1
            print(f"Step {step}: Valid action - {info.get('scheduled_job')}")
        else:
            invalid_actions += 1
            if step < 10:  # Only print first few invalid actions
                print(f"Step {step}: Invalid - {info.get('reason')}")
        
        step += 1
        done = done or truncated
    
    print(f"\nEpisode summary:")
    print(f"Total steps: {step}")
    print(f"Valid actions: {valid_actions}")
    print(f"Invalid actions: {invalid_actions}")
    print(f"Jobs scheduled: {len(env.scheduled_jobs)}")
    print(f"Jobs completed: {len(env.completed_jobs)}")
    print(f"Total families: {len(env.families)}")
    
    # Check what jobs exist
    print("\nJob families:")
    for fid, family in env.families.items():
        print(f"  {fid}: {family['total_sequences']} sequences")
else:
    print(f"\nNo trained model found at {model_path}")
    print("Run training first with: uv run python ../app_2/phase3/train_foundation.py")

# Test random actions
print("\n\nTesting with random valid actions:")
env.reset()
for i in range(10):
    valid_actions = env._get_valid_actions()
    if valid_actions:
        action = np.array(valid_actions[0])
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action {action} -> Valid: {info.get('action_valid')}, Reward: {reward:.2f}")
        if info.get('action_valid'):
            print(f"  Scheduled: {info.get('scheduled_job')}")
    else:
        print("No valid actions available!")
        break