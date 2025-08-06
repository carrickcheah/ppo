#!/usr/bin/env python
"""
Train a single stage quickly for immediate use.
This trains just on 40 jobs which is a good balance of complexity and speed.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
from src.models.rollout_buffer import RolloutBuffer
import numpy as np
from tqdm import tqdm

def train_single_stage():
    """Train on 40 jobs dataset for practical use."""
    
    print("=" * 60)
    print("Single Stage Fast Training")
    print("Training on 40 jobs dataset (127 tasks)")
    print("=" * 60)
    
    # Create environment
    env = SchedulingEnv("data/40_jobs.json", max_steps=1500)
    print(f"\nEnvironment: {env.n_tasks} tasks, {env.n_machines} machines")
    
    # Create PPO model with aggressive settings for fast learning
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=2e-3,  # High learning rate
        n_epochs=3,  # Fewer epochs for speed
        batch_size=32,  # Small batch
        device="mps"
    )
    
    # Create buffer
    buffer = RolloutBuffer(
        buffer_size=256,  # Small buffer for fast updates
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    print("\nTraining for 10,000 timesteps (should take ~2-3 minutes)...")
    
    # Training loop
    obs, info = env.reset()
    episode_rewards = []
    current_reward = 0
    
    pbar = tqdm(total=10000, desc="Training")
    timesteps = 0
    
    while timesteps < 10000:
        # Collect rollout
        for _ in range(256):
            action, pred_info = ppo.predict(obs, info['action_mask'], deterministic=False)
            next_obs, reward, done, truncated, next_info = env.step(action)
            
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=pred_info['value'],
                log_prob=pred_info['log_prob'],
                action_mask=info['action_mask'],
                done=done or truncated
            )
            
            current_reward += reward
            timesteps += 1
            
            if done or truncated:
                episode_rewards.append(current_reward)
                current_reward = 0
                obs, info = env.reset()
            else:
                obs = next_obs
                info = next_info
        
        # Get last value
        _, pred_info = ppo.predict(obs, info['action_mask'], deterministic=False)
        buffer.compute_returns_and_advantages(
            gamma=0.99,
            gae_lambda=0.95,
            last_value=pred_info['value']
        )
        
        # Update model
        ppo.train_on_buffer(buffer, None)
        buffer.reset()
        
        # Update progress
        pbar.update(256)
        if episode_rewards:
            pbar.set_postfix({'reward': f"{np.mean(episode_rewards[-5:]):.1f}"})
    
    pbar.close()
    
    # Save model
    os.makedirs("checkpoints/fast", exist_ok=True)
    ppo.save("checkpoints/fast/model_40jobs.pth")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.1f}")
    print(f"Model saved to: checkpoints/fast/model_40jobs.pth")
    print("\nYou can now use this model to schedule up to 40 job families!")
    print("=" * 60)
    
    return ppo

if __name__ == "__main__":
    train_single_stage()