"""
Test script to verify PPO model components work correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from environments.scheduling_env import SchedulingEnv
from models.ppo_scheduler import PPOScheduler
from models.rollout_buffer import RolloutBuffer


def test_ppo_components():
    """Test PPO model components."""
    print("Testing PPO Model Components")
    print("=" * 50)
    
    # Create environment
    env = SchedulingEnv("data/10_jobs.json", max_steps=100)
    print(f"✓ Environment loaded")
    
    # Create PPO model
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=3e-4
    )
    print(f"✓ PPO model created")
    print(f"  - Observation dim: {env.observation_space.shape[0]}")
    print(f"  - Action dim: {env.action_space.n}")
    print(f"  - Device: {ppo.device}")
    
    # Test prediction
    obs, info = env.reset()
    action_mask = info['action_mask']
    
    action, pred_info = ppo.predict(obs, action_mask, deterministic=False)
    print(f"\n✓ Prediction test")
    print(f"  - Action: {action}")
    print(f"  - Value: {pred_info['value']:.3f}")
    print(f"  - Log prob: {pred_info['log_prob']:.3f}")
    
    # Test rollout buffer
    buffer = RolloutBuffer(
        buffer_size=100,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    print(f"\n✓ Rollout buffer created")
    
    # Collect some experience
    print(f"\nCollecting experience...")
    for step in range(10):
        # Get action
        action, pred_info = ppo.predict(obs, action_mask, deterministic=False)
        
        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        # Add to buffer
        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            value=pred_info['value'],
            log_prob=pred_info['log_prob'],
            action_mask=action_mask,
            done=terminated or truncated
        )
        
        if terminated or truncated:
            obs, info = env.reset()
            action_mask = info['action_mask']
        else:
            obs = next_obs
            action_mask = next_info['action_mask']
            
    print(f"  - Collected {len(buffer)} steps")
    
    # Compute returns
    buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    stats = buffer.get_stats()
    print(f"\n✓ GAE computation")
    print(f"  - Mean reward: {stats['mean_reward']:.3f}")
    print(f"  - Mean advantage: {stats['mean_advantage']:.3f}")
    
    # Test training step
    print(f"\n✓ Training test")
    train_stats = ppo.train_on_buffer(buffer)
    print(f"  - Policy loss: {train_stats['policy_loss']:.3f}")
    print(f"  - Value loss: {train_stats['value_loss']:.3f}")
    print(f"  - Total loss: {train_stats['total_loss']:.3f}")
    
    # Test save/load
    checkpoint_path = "test_checkpoint.pth"
    ppo.save(checkpoint_path)
    print(f"\n✓ Model saved to {checkpoint_path}")
    
    # Create new model and load
    ppo2 = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    ppo2.load(checkpoint_path)
    print(f"✓ Model loaded successfully")
    
    # Clean up
    os.remove(checkpoint_path)
    
    print("\n" + "=" * 50)
    print("All PPO tests passed!")


if __name__ == "__main__":
    test_ppo_components()