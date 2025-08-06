#!/usr/bin/env python
"""
Simple training script for 100-job dataset.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler

def train_100_jobs():
    """Train on 100-job dataset."""
    
    print("=" * 60)
    print("TRAINING PPO MODEL - 100 JOBS")
    print("=" * 60)
    
    # Setup
    data_path = 'data/100_jobs.json'
    checkpoint_dir = 'checkpoints/100jobs'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create environment
    env = SchedulingEnv(data_path, max_steps=5000)
    print(f"\nEnvironment: {env.n_tasks} tasks, {env.n_machines} machines")
    
    # Create model with larger network
    model = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=512,
        device='mps'
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Training parameters
    n_episodes = 500
    batch_size = 256
    gamma = 0.99
    clip_epsilon = 0.2
    
    print(f"\nTraining for {n_episodes} episodes...")
    print("-" * 60)
    
    best_completion = 0
    
    for episode in range(n_episodes):
        # Collect trajectory
        obs, info = env.reset()
        done = False
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        masks = []
        
        episode_reward = 0
        steps = 0
        
        while not done and steps < 5000:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to('mps')
            mask_tensor = torch.FloatTensor(info['action_mask']).unsqueeze(0).to('mps')
            
            # Get action
            with torch.no_grad():
                action_probs = model.policy(obs_tensor)
                action_probs = action_probs * mask_tensor
                
                if action_probs.sum() > 0:
                    action_probs = action_probs / action_probs.sum()
                else:
                    action_probs = mask_tensor / mask_tensor.sum()
                
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = model.value(obs_tensor)
            
            # Store experience
            states.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob)
            values.append(value)
            masks.append(info['action_mask'])
            
            # Take action
            next_obs, reward, terminated, truncated, next_info = env.step(action.item())
            rewards.append(reward)
            episode_reward += reward
            
            done = terminated or truncated
            obs = next_obs
            info = next_info
            steps += 1
        
        completion_rate = info['tasks_scheduled'] / info['total_tasks']
        
        # Update policy if we have experience
        if len(states) > 0 and episode > 0 and episode % 5 == 0:
            # Convert to tensors
            states_tensor = torch.FloatTensor(np.array(states)).to('mps')
            actions_tensor = torch.LongTensor(actions).to('mps')
            old_log_probs = torch.stack(log_probs).detach()
            values_tensor = torch.stack(values).squeeze().detach()
            masks_tensor = torch.FloatTensor(np.array(masks)).to('mps')
            
            # Calculate returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to('mps')
            advantages = (returns - values_tensor).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(4):
                # Get current predictions
                action_probs = model.policy(states_tensor)
                action_probs = action_probs * masks_tensor
                action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-8)
                
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions_tensor)
                new_values = model.value(states_tensor).squeeze()
                
                # Calculate losses
                ratio = torch.exp(new_log_probs - old_log_probs.squeeze())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values, returns)
                entropy = dist.entropy().mean()
                
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        
        # Progress
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1:3d} | Completion: {completion_rate*100:5.1f}% "
                  f"({info['tasks_scheduled']}/{info['total_tasks']}) | Reward: {episode_reward:7.1f}")
        
        # Save best
        if completion_rate > best_completion:
            best_completion = completion_rate
            model.save(os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  → New best: {best_completion*100:.1f}%")
    
    # Save final
    model.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best: {best_completion*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    train_100_jobs()