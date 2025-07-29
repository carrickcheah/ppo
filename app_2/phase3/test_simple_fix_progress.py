"""
Test simple fix training progress
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


def test_checkpoint(checkpoint_path, stage_name='toy_normal'):
    """Test a checkpoint."""
    
    # Load model
    model = PPO.load(checkpoint_path)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = Monitor(env)
    
    # Test for 10 episodes
    results = []
    
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < env.env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            steps += 1
        
        scheduled = len(env.env.scheduled_jobs)
        total = env.env.total_tasks
        rate = scheduled / total
        results.append(rate)
        
        if ep < 3:  # Show first 3
            print(f"  Episode {ep+1}: {scheduled}/{total} = {rate:.1%}")
    
    avg_rate = np.mean(results)
    return avg_rate


def main():
    """Test progression of simple fix training."""
    
    print("SIMPLE FIX TRAINING PROGRESS")
    print("=" * 60)
    print("Testing checkpoints to see if we're approaching 80%...")
    print("-" * 60)
    
    checkpoints = [
        (100000, "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/simple_fix/toy_normal/checkpoint_100000_steps.zip"),
        (300000, "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/simple_fix/toy_normal/checkpoint_300000_steps.zip"),
        (500000, "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/simple_fix/toy_normal/checkpoint_500000_steps.zip"),
        (700000, "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/simple_fix/toy_normal/checkpoint_700000_steps.zip")
    ]
    
    print("\nBaseline (with -5 penalty): 56.2%")
    print("Target: 80%")
    print("\nWith -2 penalty:")
    
    for steps, path in checkpoints:
        if os.path.exists(path):
            print(f"\n{steps:,} steps:")
            rate = test_checkpoint(path)
            
            improvement = rate - 0.562
            status = "✓ TARGET ACHIEVED!" if rate >= 0.8 else ""
            print(f"Average: {rate:.1%} (improved {improvement:+.1%}) {status}")
    
    # Final summary
    print("\n" + "-"*60)
    print("CONCLUSION:")
    
    if rate >= 0.8:
        print(f"✓ Simple fix WORKED! Just reducing penalty from -5 to -2 achieved {rate:.1%}!")
        print("✓ No complex wrappers needed!")
        print("✓ No action masking needed!")
        print("✓ Just a simple penalty adjustment!")
    elif rate >= 0.7:
        print(f"Close! Achieved {rate:.1%}. May need full 1M steps or penalty of -1.5")
    else:
        print(f"Progress made ({rate:.1%}) but may need further penalty reduction")


if __name__ == "__main__":
    main()