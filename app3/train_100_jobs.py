#!/usr/bin/env python
"""
Train PPO model on 100-job dataset for scaling up.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
from src.training.ppo_trainer import PPOTrainer

def train_100_jobs():
    """Train model on 100-job dataset."""
    
    print("=" * 60)
    print("TRAINING PPO MODEL - 100 JOBS SCALE")
    print("=" * 60)
    
    # Configuration for 100 jobs
    config = {
        'data_path': 'data/100_jobs.json',
        'max_steps': 5000,  # More steps for larger problem
        'device': 'mps',
        'learning_rate': 3e-4,
        'batch_size': 256,  # Larger batch for stability
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'checkpoint_dir': 'checkpoints/100jobs',
        'total_timesteps': 200000,  # More training for larger scale
    }
    
    # Create environment
    print("\nInitializing environment...")
    env = SchedulingEnv(config['data_path'], max_steps=config['max_steps'])
    print(f"Environment: {env.n_tasks} tasks, {env.n_machines} machines")
    
    # Create model
    print("\nInitializing PPO model...")
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=512,  # Larger network for more complex problem
        device=config['device']
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = PPOTrainer(
        env=env,
        model=ppo,
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        value_coef=config['value_coef'],
        entropy_coef=config['entropy_coef'],
        max_grad_norm=config['max_grad_norm'],
        device=config['device']
    )
    
    # Training loop
    print(f"\nStarting training for {config['total_timesteps']} timesteps...")
    print("-" * 60)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    best_completion = 0
    episode = 0
    timesteps = 0
    
    while timesteps < config['total_timesteps']:
        episode += 1
        
        # Collect experience
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        action_masks = []
        
        while not done and episode_steps < config['max_steps']:
            # Get action from model
            with torch.no_grad():
                action, log_prob, value = ppo.get_action_and_value(obs, info['action_mask'])
            
            # Store experience
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            action_masks.append(info['action_mask'])
            
            # Take action
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            
            obs = next_obs
            info = next_info
            episode_steps += 1
            timesteps += 1
        
        # Calculate completion rate
        completion_rate = info['tasks_scheduled'] / info['total_tasks']
        
        # Train on collected experience
        if len(observations) > 0:
            # Get final value for bootstrapping
            with torch.no_grad():
                _, _, next_value = ppo.get_action_and_value(obs, info['action_mask'])
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(np.array(observations)).to(config['device'])
            actions_tensor = torch.LongTensor(actions).to(config['device'])
            log_probs_tensor = torch.stack(log_probs)
            values_tensor = torch.stack(values).squeeze()
            rewards_tensor = torch.FloatTensor(rewards).to(config['device'])
            dones_tensor = torch.FloatTensor(dones).to(config['device'])
            masks_tensor = torch.FloatTensor(np.array(action_masks)).to(config['device'])
            
            # Calculate advantages
            advantages = torch.zeros_like(rewards_tensor)
            returns = torch.zeros_like(rewards_tensor)
            
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value_t = next_value
                else:
                    next_value_t = values_tensor[t + 1]
                
                delta = rewards_tensor[t] + config['gamma'] * next_value_t * (1 - dones_tensor[t]) - values_tensor[t]
                gae = delta + config['gamma'] * config['gae_lambda'] * (1 - dones_tensor[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values_tensor[t]
            
            # Update policy
            policy_loss, value_loss = trainer.update(
                obs_tensor, actions_tensor, log_probs_tensor,
                values_tensor, returns, advantages, masks_tensor
            )
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Steps: {timesteps:6d} | "
                  f"Completion: {completion_rate*100:5.1f}% ({info['tasks_scheduled']}/{info['total_tasks']}) | "
                  f"Reward: {episode_reward:7.1f}")
        
        # Save best model
        if completion_rate > best_completion:
            best_completion = completion_rate
            model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            ppo.save(model_path)
            print(f"  → New best model saved! Completion: {best_completion*100:.1f}%")
        
        # Save periodic checkpoint
        if episode % 100 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_ep{episode}.pth')
            ppo.save(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    ppo.save(final_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best completion rate: {best_completion*100:.1f}%")
    print(f"Models saved in: {config['checkpoint_dir']}")
    print("=" * 60)
    
    return best_completion

if __name__ == "__main__":
    train_100_jobs()