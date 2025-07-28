"""
Test progression of checkpoints to see if model improved
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed
from phase3.environments.action_masked_env import ToyStageActionMasker
from stable_baselines3.common.monitor import Monitor


def test_checkpoint(stage_name: str, checkpoint_steps: int):
    """Test a specific checkpoint."""
    checkpoint_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/masked/{stage_name}/masked_checkpoint_{checkpoint_steps}_steps.zip"
    
    if not os.path.exists(checkpoint_path):
        return None
    
    # Load model
    model = MaskablePPO.load(checkpoint_path)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = ToyStageActionMasker(env)
    env = Monitor(env)
    
    # Test for 5 episodes
    total_scheduled = 0
    total_possible = 0
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            steps += 1
        
        original_env = env.env.original_env
        scheduled = len(original_env.scheduled_jobs) if hasattr(original_env, 'scheduled_jobs') else 0
        total = original_env.total_tasks if hasattr(original_env, 'total_tasks') else 1
        
        total_scheduled += scheduled
        total_possible += total
    
    return total_scheduled / total_possible if total_possible > 0 else 0


if __name__ == "__main__":
    print("Testing Checkpoint Progression for toy_normal")
    print("=" * 50)
    
    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
    
    for steps in checkpoints:
        rate = test_checkpoint('toy_normal', steps)
        if rate is not None:
            print(f"{steps:>6} steps: {rate:.1%}")
    
    print("\nConclusion:")
    print("If performance doesn't improve, the issue might be:")
    print("1. Action masking is working but rewards are still problematic")
    print("2. The flattened action space is harder to learn")
    print("3. Need different hyperparameters for masked training")