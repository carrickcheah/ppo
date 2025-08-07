#!/usr/bin/env python
"""
Advanced training script with optimized hyperparameters and techniques.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import json
from datetime import datetime

def train_advanced_model():
    """Train with advanced techniques for better performance."""
    
    print("="*60)
    print("ADVANCED PPO TRAINING")
    print("="*60)
    
    # STRATEGY 1: Better hyperparameters
    hyperparameters = {
        # Increased model capacity
        'hidden_dim': 512,  # Larger network (was 256)
        'n_layers': 4,      # Deeper network (was 3)
        
        # Optimized PPO parameters
        'learning_rate': 1e-4,  # Lower LR for stability
        'gamma': 0.995,         # Higher discount for long-term planning
        'epsilon': 0.1,         # Lower clip range for stability
        'value_coef': 0.5,      # Standard value loss weight
        'entropy_coef': 0.005,  # Lower entropy for exploitation
        
        # Training parameters
        'n_episodes': 3000,     # More episodes (was 1000)
        'n_epochs': 10,         # More epochs per update (was 5)
        'batch_size': 256,      # Larger batch (was 128)
        'update_frequency': 20,  # Update every 20 episodes
        'gradient_clip': 0.5,   # Gradient clipping for stability
    }
    
    # STRATEGY 2: Curriculum learning (start easy, increase difficulty)
    curriculum_stages = [
        {'name': 'Stage 1: 40 jobs', 'data': 'data/40_jobs.json', 'episodes': 500},
        {'name': 'Stage 2: 60 jobs', 'data': 'data/60_jobs.json', 'episodes': 700},
        {'name': 'Stage 3: 80 jobs', 'data': 'data/80_jobs.json', 'episodes': 800},
        {'name': 'Stage 4: 100 jobs', 'data': 'data/100_jobs.json', 'episodes': 1000},
    ]
    
    # Initialize model with larger capacity
    initial_env = SchedulingEnv('data/40_jobs.json', max_steps=5000)
    model = PPOScheduler(
        obs_dim=initial_env.observation_space.shape[0],
        action_dim=initial_env.action_space.n,
        hidden_dim=hyperparameters['hidden_dim'],
        n_layers=hyperparameters['n_layers'],
        lr=hyperparameters['learning_rate'],
        gamma=hyperparameters['gamma'],
        epsilon=hyperparameters['epsilon'],
        value_coef=hyperparameters['value_coef'],
        entropy_coef=hyperparameters['entropy_coef'],
        device='mps'
    )
    
    print("\nModel architecture:")
    print(f"  Hidden dimensions: {hyperparameters['hidden_dim']}")
    print(f"  Number of layers: {hyperparameters['n_layers']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # STRATEGY 3: Experience replay buffer
    class ExperienceBuffer:
        def __init__(self, capacity=10000):
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []
            self.masks = []
            self.capacity = capacity
            
        def add(self, state, action, reward, log_prob, value, mask):
            if len(self.states) >= self.capacity:
                # Remove oldest
                self.states.pop(0)
                self.actions.pop(0)
                self.rewards.pop(0)
                self.log_probs.pop(0)
                self.values.pop(0)
                self.masks.pop(0)
            
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.masks.append(mask)
        
        def sample(self, batch_size):
            indices = np.random.choice(len(self.states), min(batch_size, len(self.states)), replace=False)
            return (
                [self.states[i] for i in indices],
                [self.actions[i] for i in indices],
                [self.rewards[i] for i in indices],
                [self.log_probs[i] for i in indices],
                [self.values[i] for i in indices],
                [self.masks[i] for i in indices]
            )
        
        def clear(self):
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.values.clear()
            self.masks.clear()
    
    buffer = ExperienceBuffer(capacity=10000)
    
    # Training with curriculum
    total_episodes = 0
    best_reward = -float('inf')
    
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n{'='*60}")
        print(f"{stage['name']}")
        print(f"{'='*60}")
        
        # Create environment for this stage
        env = SchedulingEnv(stage['data'], max_steps=5000)
        
        # Adjust learning rate with decay
        current_lr = hyperparameters['learning_rate'] * (0.9 ** stage_idx)
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"Learning rate: {current_lr:.6f}")
        
        stage_best_completion = 0
        stage_rewards = []
        
        for episode in range(stage['episodes']):
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
            
            # STRATEGY 4: Exploration decay
            exploration_rate = max(0.1, 1.0 - (total_episodes / 2000))
            
            while not done and steps < 5000:
                # Get action with exploration
                if np.random.random() < exploration_rate * 0.1:  # 10% random exploration
                    valid_actions = np.where(info['action_mask'])[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        action = 0
                    
                    # Still need to get log_prob and value
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to('mps')
                    mask_tensor = torch.BoolTensor(info['action_mask']).unsqueeze(0).to('mps')
                    with torch.no_grad():
                        logits, value, _ = model.policy(obs_tensor, mask_tensor)
                        logits = logits.masked_fill(mask_tensor == 0, -1e8)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(torch.tensor(action).to('mps'))
                else:
                    with torch.no_grad():
                        action, _ = model.predict(obs, info['action_mask'], deterministic=False)
                        
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to('mps')
                        mask_tensor = torch.BoolTensor(info['action_mask']).unsqueeze(0).to('mps')
                        
                        logits, value, _ = model.policy(obs_tensor, mask_tensor)
                        logits = logits.masked_fill(mask_tensor == 0, -1e8)
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(torch.tensor(action).to('mps'))
                
                # Store experience
                states.append(obs)
                actions.append(action)
                masks.append(info['action_mask'])
                log_probs.append(log_prob)
                values.append(value)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # STRATEGY 5: Reward shaping
                shaped_reward = reward
                if info['tasks_scheduled'] > 0:
                    # Bonus for scheduling efficiency
                    completion_rate = info['tasks_scheduled'] / info['total_tasks']
                    shaped_reward += completion_rate * 0.1
                    
                    # Penalty for late jobs
                    if 'late_percentage' in info:
                        shaped_reward -= info['late_percentage'] * 0.2
                
                rewards.append(shaped_reward)
                episode_reward += shaped_reward
                steps += 1
            
            # Add to buffer
            for i in range(len(states)):
                buffer.add(states[i], actions[i], rewards[i], 
                          log_probs[i], values[i], masks[i])
            
            # Update model
            if (episode + 1) % hyperparameters['update_frequency'] == 0 and len(buffer.states) > hyperparameters['batch_size']:
                # Sample from buffer
                sampled = buffer.sample(hyperparameters['batch_size'] * 2)
                
                # Train on sampled experience
                loss = model.train_on_batch(
                    sampled[0], sampled[1], sampled[2],
                    sampled[3], sampled[4], sampled[5],
                    n_epochs=hyperparameters['n_epochs']
                )
            
            # Track progress
            completion_rate = info['tasks_scheduled'] / info['total_tasks']
            stage_rewards.append(episode_reward)
            
            if completion_rate > stage_best_completion:
                stage_best_completion = completion_rate
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    # Save best model
                    os.makedirs('checkpoints/advanced', exist_ok=True)
                    model.save('checkpoints/advanced/best_model.pth')
            
            # Logging
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(stage_rewards[-50:])
                print(f"Episode {episode+1}/{stage['episodes']}: "
                      f"Completion: {completion_rate:.1%}, "
                      f"Reward: {episode_reward:.1f}, "
                      f"Avg(50): {avg_reward:.1f}, "
                      f"Best: {stage_best_completion:.1%}")
            
            total_episodes += 1
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test on 100 jobs
    test_env = SchedulingEnv('data/100_jobs.json', max_steps=5000)
    obs, info = test_env.reset()
    done = False
    steps = 0
    
    while not done and steps < 5000:
        action, _ = model.predict(obs, info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        steps += 1
    
    print(f"\nFinal Results:")
    print(f"  Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']} ({info['tasks_scheduled']/info['total_tasks']*100:.1f}%)")
    print(f"  Total reward: {reward:.2f}")
    
    # Save final model
    model.save('checkpoints/advanced/final_model.pth')
    print(f"\nModels saved to checkpoints/advanced/")
    
    # Save training config
    config = {
        'hyperparameters': hyperparameters,
        'curriculum_stages': curriculum_stages,
        'final_performance': {
            'completion_rate': info['tasks_scheduled'] / info['total_tasks'],
            'total_reward': float(reward),
            'best_reward': float(best_reward)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('checkpoints/advanced/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining complete!")
    print("="*60)

if __name__ == "__main__":
    train_advanced_model()