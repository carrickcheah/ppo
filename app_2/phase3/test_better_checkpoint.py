"""
Test the better reward checkpoint
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from phase3.environments.better_reward_env import BetterRewardEnvironment


def test_checkpoint(checkpoint_path):
    """Test a checkpoint."""
    
    # Load model
    model = PPO.load(checkpoint_path)
    
    # Create environment
    env = BetterRewardEnvironment('toy_normal', verbose=False)
    env = Monitor(env)
    
    # Test for 10 episodes
    results = []
    
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            done = done or truncated
            steps += 1
            if steps > 200:  # Safety limit
                break
        
        scheduled = len(env.env.scheduled_jobs)
        total = env.env.total_tasks
        rate = scheduled / total
        
        results.append(rate)
        print(f"Episode {ep+1}: {scheduled}/{total} = {rate:.1%}, Reward: {ep_reward:.1f}, Steps: {steps}")
    
    avg_rate = np.mean(results)
    print(f"\nAverage completion: {avg_rate:.1%}")
    
    if avg_rate >= 0.8:
        print("✓ ACHIEVED 80% TARGET!")
    else:
        print(f"✗ Still short of 80% target")
    
    return avg_rate


if __name__ == "__main__":
    checkpoints = [
        ("/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/better_rewards/toy_normal/checkpoint_50000_steps.zip", 50000),
        ("/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/better_rewards/toy_normal/checkpoint_100000_steps.zip", 100000),
        ("/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/better_rewards/toy_normal/checkpoint_150000_steps.zip", 150000)
    ]
    
    print("Testing Better Reward Checkpoints")
    print("=" * 50)
    
    for path, steps in checkpoints:
        print(f"\nCheckpoint at {steps} steps:")
        print("-" * 30)
        test_checkpoint(path)