"""
Quick test of the masked model to see if it achieved 80%
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed
from phase3.environments.action_masked_env import ToyStageActionMasker
from stable_baselines3.common.monitor import Monitor


def test_masked_model(stage_name: str):
    """Test the trained masked model."""
    # Try checkpoint first
    checkpoint_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/masked/{stage_name}/masked_checkpoint_500000_steps.zip"
    model_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/models_masked/{stage_name}_masked.zip"
    
    if os.path.exists(checkpoint_path):
        model_path = checkpoint_path
        print(f"Using checkpoint: {checkpoint_path}")
    elif not os.path.exists(model_path):
        print(f"Model not found")
        return
    
    # Load model
    model = MaskablePPO.load(model_path)
    
    # Create environment with masking
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = ToyStageActionMasker(env)
    env = Monitor(env)
    
    # Test for 10 episodes
    total_scheduled = 0
    total_possible = 0
    
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            steps += 1
        
        # Get original env
        original_env = env.env.original_env
        scheduled = len(original_env.scheduled_jobs) if hasattr(original_env, 'scheduled_jobs') else 0
        total = original_env.total_tasks if hasattr(original_env, 'total_tasks') else 1
        rate = scheduled / total
        
        print(f"Episode {ep+1}: {scheduled}/{total} = {rate:.1%}")
        
        total_scheduled += scheduled
        total_possible += total
    
    avg_rate = total_scheduled / total_possible
    print(f"\nAverage: {avg_rate:.1%}")
    
    if avg_rate >= 0.8:
        print(f"✓ {stage_name} ACHIEVED 80% TARGET!")
    else:
        print(f"✗ {stage_name} fell short of 80% target")
    
    return avg_rate


if __name__ == "__main__":
    print("Testing Masked Models")
    print("=" * 50)
    
    for stage in ['toy_normal', 'toy_hard', 'toy_multi']:
        print(f"\n{stage}:")
        test_masked_model(stage)