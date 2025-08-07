#!/usr/bin/env python
"""
10X Advanced PPO Training Script
Makes the model significantly smarter through:
1. Larger network (512->512->256->128)
2. Better observations and rewards
3. Curriculum learning
4. Extended training (10,000 episodes)
5. Smart exploration
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
import math

def cosine_lr_schedule(initial_lr, current_step, total_steps):
    """Cosine learning rate decay."""
    return initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / total_steps))

def train_10x_model():
    """Train PPO model with 10x improvements."""
    
    print("="*60)
    print("10X ADVANCED PPO TRAINING")
    print("="*60)
    
    # Training configuration
    config = {
        # Network architecture (4x bigger)
        'hidden_sizes': (512, 512, 256, 128),
        'dropout_rate': 0.1,
        'use_batch_norm': True,
        
        # PPO hyperparameters
        'learning_rate': 3e-4,
        'min_learning_rate': 1e-5,
        'n_epochs': 10,
        'batch_size': 256,
        'clip_range': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        
        # Exploration
        'initial_exploration': 0.1,  # 10% random actions initially
        'final_exploration': 0.01,   # 1% at the end
        
        # Training schedule
        'total_episodes': 10000,
        'save_frequency': 500,
        'eval_frequency': 100,
        
        # Curriculum learning stages
        'curriculum': [
            {'name': 'Stage 1: Easy', 'data': 'data/40_jobs.json', 'episodes': 2000},
            {'name': 'Stage 2: Medium', 'data': 'data/60_jobs.json', 'episodes': 3000},
            {'name': 'Stage 3: Hard', 'data': 'data/80_jobs.json', 'episodes': 3000},
            {'name': 'Stage 4: Expert', 'data': 'data/100_jobs.json', 'episodes': 2000},
        ]
    }
    
    # Initialize environment and model
    initial_env = SchedulingEnv('data/40_jobs.json', max_steps=5000)
    
    model = PPOScheduler(
        obs_dim=initial_env.observation_space.shape[0],
        action_dim=initial_env.action_space.n,
        learning_rate=config['learning_rate'],
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        clip_range=config['clip_range'],
        value_loss_coef=config['value_loss_coef'],
        entropy_coef=config['entropy_coef'],
        max_grad_norm=config['max_grad_norm'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate'],
        use_batch_norm=config['use_batch_norm'],
        exploration_rate=config['initial_exploration'],
        device='mps'
    )
    
    # Set training mode
    if hasattr(model, 'set_training_mode'):
        model.set_training_mode(True)
    
    print("\nModel Configuration:")
    print(f"  Network architecture: {config['hidden_sizes']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"  Device: {model.device}")
    
    # Training variables
    total_episodes = 0
    best_reward = -float('inf')
    best_completion_rate = 0
    all_rewards = []
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints/10x'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Curriculum learning loop
    for stage_idx, stage in enumerate(config['curriculum']):
        print(f"\n{'='*60}")
        print(f"{stage['name']}")
        print(f"Data: {stage['data']}, Episodes: {stage['episodes']}")
        print(f"{'='*60}")
        
        # Create environment for this stage
        env = SchedulingEnv(stage['data'], max_steps=5000)
        
        # Adjust learning rate with cosine decay
        progress = total_episodes / config['total_episodes']
        current_lr = cosine_lr_schedule(
            config['learning_rate'],
            total_episodes,
            config['total_episodes']
        )
        current_lr = max(current_lr, config['min_learning_rate'])
        
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Adjust exploration rate
        exploration_decay = (config['initial_exploration'] - config['final_exploration']) * (1 - progress)
        current_exploration = config['final_exploration'] + exploration_decay
        if hasattr(model, 'exploration_rate'):
            model.exploration_rate = current_exploration
        
        print(f"Learning rate: {current_lr:.6f}")
        print(f"Exploration rate: {current_exploration:.3f}")
        
        stage_rewards = []
        stage_completions = []
        
        # Training episodes for this stage
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
            
            while not done and steps < 5000:
                # Get action from model
                with torch.no_grad():
                    action, action_info = model.predict(obs, info['action_mask'], deterministic=False)
                    
                    # Get log_prob and value for training
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
                    mask_tensor = torch.BoolTensor(info['action_mask']).unsqueeze(0).to(model.device)
                    
                    _, _, dist = model.policy(obs_tensor, mask_tensor)
                    log_prob = dist.log_prob(torch.tensor(action).to(model.device))
                    value = model.policy(obs_tensor, mask_tensor)[1]
                
                # Store experience
                states.append(obs)
                actions.append(action)
                masks.append(info['action_mask'])
                log_probs.append(log_prob)
                values.append(value)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Enhanced reward shaping
                shaped_reward = reward
                
                # Progress bonus
                completion_ratio = info['tasks_scheduled'] / info['total_tasks']
                shaped_reward += 0.1 * completion_ratio
                
                # Early completion bonus
                if done and completion_ratio == 1.0:
                    shaped_reward += 5.0  # Big bonus for 100% completion
                
                # Penalty for very late jobs
                if 'late_percentage' in info and info['late_percentage'] > 0.1:
                    shaped_reward -= info['late_percentage'] * 2.0
                
                # Efficiency bonus (minimize makespan)
                if done:
                    max_possible_time = sum([t.processing_time for t in env.loader.tasks])
                    actual_time = env.current_time
                    efficiency = max_possible_time / max(actual_time, 1.0)
                    shaped_reward += efficiency * 2.0
                
                rewards.append(shaped_reward)
                episode_reward += shaped_reward
                steps += 1
            
            # Track metrics
            completion_rate = info['tasks_scheduled'] / info['total_tasks']
            stage_rewards.append(episode_reward)
            stage_completions.append(completion_rate)
            
            # Update best model
            if completion_rate > best_completion_rate or \
               (completion_rate == best_completion_rate and episode_reward > best_reward):
                best_completion_rate = completion_rate
                best_reward = episode_reward
                model.save(f'{checkpoint_dir}/best_model.pth')
                print(f"  New best model! Completion: {completion_rate:.1%}, Reward: {episode_reward:.1f}")
            
            # Train on collected experience
            if len(states) > 0:
                # Convert to tensors
                states_t = torch.FloatTensor(np.array(states)).to(model.device)
                actions_t = torch.LongTensor(actions).to(model.device)
                rewards_t = torch.FloatTensor(rewards).to(model.device)
                masks_t = torch.BoolTensor(np.array(masks)).to(model.device)
                
                # Stack log_probs and values
                if len(log_probs) > 0:
                    old_log_probs = torch.stack(log_probs).squeeze()
                    old_values = torch.stack(values).squeeze()
                else:
                    continue
                
                # Compute advantages
                advantages = torch.zeros_like(rewards_t)
                last_advantage = 0
                last_value = 0
                
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_value = 0
                    else:
                        next_value = old_values[t + 1]
                    
                    delta = rewards_t[t] + model.gamma * next_value - old_values[t]
                    advantages[t] = last_advantage = delta + model.gamma * model.gae_lambda * last_advantage
                
                returns = advantages + old_values.detach()
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO update
                for _ in range(model.n_epochs):
                    # Mini-batch updates
                    batch_size = min(model.batch_size, len(states))
                    indices = np.random.permutation(len(states))
                    
                    for start in range(0, len(states), batch_size):
                        end = start + batch_size
                        batch_indices = indices[start:end]
                        
                        batch_states = states_t[batch_indices]
                        batch_actions = actions_t[batch_indices]
                        batch_masks = masks_t[batch_indices]
                        batch_old_log_probs = old_log_probs[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        batch_returns = returns[batch_indices]
                        
                        # Get current policy outputs
                        logits, values, dist = model.policy(batch_states, batch_masks)
                        log_probs = dist.log_prob(batch_actions)
                        
                        # PPO loss
                        ratio = torch.exp(log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - model.clip_range, 1 + model.clip_range) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = model.value_loss_coef * torch.nn.functional.mse_loss(values.squeeze(), batch_returns)
                        
                        # Entropy bonus
                        entropy_loss = -model.entropy_coef * dist.entropy().mean()
                        
                        # Total loss
                        loss = policy_loss + value_loss + entropy_loss
                        
                        # Optimize
                        model.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                        model.optimizer.step()
            
            # Logging
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(stage_rewards[-50:])
                avg_completion = np.mean(stage_completions[-50:])
                print(f"Episode {episode+1}/{stage['episodes']}: "
                      f"Completion: {completion_rate:.1%}, "
                      f"Reward: {episode_reward:.1f}, "
                      f"Avg(50): R={avg_reward:.1f}, C={avg_completion:.1%}")
            
            # Save checkpoint
            if (total_episodes + 1) % config['save_frequency'] == 0:
                model.save(f'{checkpoint_dir}/checkpoint_{total_episodes+1}.pth')
                print(f"  Saved checkpoint at episode {total_episodes+1}")
            
            total_episodes += 1
            all_rewards.append(episode_reward)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test on largest dataset
    model.set_training_mode(False) if hasattr(model, 'set_training_mode') else None
    model.exploration_rate = 0  # No exploration during evaluation
    
    test_env = SchedulingEnv('data/100_jobs.json', max_steps=5000)
    obs, info = test_env.reset()
    done = False
    steps = 0
    
    while not done and steps < 5000:
        action, _ = model.predict(obs, info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        steps += 1
    
    print(f"\nFinal Performance:")
    print(f"  Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']} "
          f"({info['tasks_scheduled']/info['total_tasks']*100:.1f}%)")
    print(f"  Total reward: {reward:.2f}")
    print(f"  Steps taken: {steps}")
    
    # Save final model
    model.save(f'{checkpoint_dir}/final_model.pth')
    
    # Save training history
    history = {
        'config': config,
        'total_episodes': total_episodes,
        'best_reward': float(best_reward),
        'best_completion_rate': float(best_completion_rate),
        'final_performance': {
            'completion_rate': info['tasks_scheduled'] / info['total_tasks'],
            'reward': float(reward),
            'steps': steps
        },
        'rewards_history': all_rewards[-1000:],  # Last 1000 episodes
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f'{checkpoint_dir}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModels saved to {checkpoint_dir}/")
    print("  - best_model.pth: Best performing model")
    print("  - final_model.pth: Final trained model")
    print("  - training_history.json: Training metrics")
    
    print("\n" + "="*60)
    print("10X TRAINING COMPLETE!")
    print("="*60)
    
    return model

if __name__ == "__main__":
    train_10x_model()