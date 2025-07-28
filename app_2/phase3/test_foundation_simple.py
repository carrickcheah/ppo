"""
Simple test of foundation models to understand observation space
"""

import os
import sys
import json
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

print("Checking Foundation Models and Environments")
print("="*60)

# Check toy_easy
stage = 'toy_easy'
print(f"\n{stage}:")
print("-"*40)

# Check model
model_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/{stage}/final_model.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path)
    print(f"Model observation space: {model.observation_space}")
    print(f"Model action space: {model.action_space}")
else:
    print("Model not found")

# Check environment
env = CurriculumEnvironmentTrulyFixed(stage, verbose=False)
print(f"\nEnvironment observation space: {env.observation_space}")
print(f"Environment action space: {env.action_space}")

# Check data file
data_path = f"/Users/carrickcheah/Project/ppo/app_2/data/stage_{stage}_clean_data.json"
if os.path.exists(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"\nData file stats:")
    print(f"  Families: {len(data['families'])}")
    print(f"  Total tasks: {sum(len(fam['tasks']) for fam in data['families'])}")
    print(f"  Machines: {len(data['machines'])}")
    
# Test observation
obs, _ = env.reset()
print(f"\nActual observation shape: {obs.shape}")
print(f"Observation sample: {obs[:10]}...")

# Try with the foundation environment that might match
try:
    from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal
    print(f"\n\nTrying CurriculumEnvironmentReal for {stage}:")
    print("-"*40)
    
    # Temporarily rename file to match expected pattern
    clean_path = f"/Users/carrickcheah/Project/ppo/app_2/data/stage_{stage}_clean_data.json"
    real_path = f"/Users/carrickcheah/Project/ppo/app_2/data/stage_{stage}_real_data.json"
    
    import shutil
    if os.path.exists(clean_path) and not os.path.exists(real_path):
        shutil.copy(clean_path, real_path)
        print(f"Copied {clean_path} to {real_path}")
    
    env_real = CurriculumEnvironmentReal(stage_name=stage, verbose=False)
    print(f"Real env observation space: {env_real.observation_space}")
    print(f"Real env action space: {env_real.action_space}")
    
    obs_real, _ = env_real.reset()
    print(f"Real env observation shape: {obs_real.shape}")
    
except Exception as e:
    print(f"Error with CurriculumEnvironmentReal: {e}")

print("\n" + "="*60)