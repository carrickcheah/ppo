#!/usr/bin/env python
"""
Improved training with longer episodes for better completion.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
from src.models.rollout_buffer import RolloutBuffer
import numpy as np
from tqdm import tqdm

def train_improved():
    """Train with longer episodes and more steps."""
    
    print("=" * 60)
    print("IMPROVED PPO TRAINING")
    print("Longer episodes for better sequence handling")
    print("=" * 60)
    
    # Create environment with MUCH longer episodes
    env = SchedulingEnv("data/40_jobs.json", max_steps=3000)  # Double the episode length
    print(f"\nEnvironment: {env.n_tasks} tasks, {env.n_machines} machines")
    print("Max steps increased to 3000 for better completion")
    
    # Create PPO model
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        n_epochs=4,
        batch_size=64,
        device="mps"
    )
    
    # Create buffer
    buffer = RolloutBuffer(
        buffer_size=512,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    print("\nTraining for 20,000 timesteps...")
    
    # Training loop
    obs, info = env.reset()
    episode_rewards = []
    episode_completions = []
    current_reward = 0
    
    pbar = tqdm(total=20000, desc="Training")
    timesteps = 0
    
    best_completion = 0
    
    while timesteps < 20000:
        # Collect rollout
        for _ in range(512):
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
                completion = next_info['tasks_scheduled'] / next_info['total_tasks']
                episode_rewards.append(current_reward)
                episode_completions.append(completion)
                
                if completion > best_completion:
                    best_completion = completion
                    ppo.save("checkpoints/fast/best_model.pth")
                
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
        pbar.update(512)
        if episode_completions:
            avg_completion = np.mean(episode_completions[-5:])
            avg_reward = np.mean(episode_rewards[-5:])
            pbar.set_postfix({
                'completion': f"{avg_completion:.1%}",
                'reward': f"{avg_reward:.0f}",
                'best': f"{best_completion:.1%}"
            })
    
    pbar.close()
    
    # Save final model
    ppo.save("checkpoints/fast/model_40jobs.pth")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best completion rate: {best_completion:.1%}")
    print(f"Final average completion: {np.mean(episode_completions[-10:]):.1%}")
    print(f"Final average reward: {np.mean(episode_rewards[-10:]):.1f}")
    print(f"Models saved to: checkpoints/fast/")
    print("=" * 60)

if __name__ == "__main__":
    train_improved()