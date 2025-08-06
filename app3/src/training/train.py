"""
Main training script for PPO scheduling.
Handles episode collection, training, and checkpointing.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.scheduling_env import SchedulingEnv
from models.ppo_scheduler import PPOScheduler, PPOConfig
from models.rollout_buffer import RolloutBuffer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_rollouts(
    env: SchedulingEnv,
    ppo: PPOScheduler,
    buffer: RolloutBuffer,
    n_steps: int
) -> Dict:
    """
    Collect rollouts for training.
    
    Args:
        env: Environment
        ppo: PPO model
        buffer: Rollout buffer
        n_steps: Number of steps to collect
        
    Returns:
        Rollout statistics
    """
    obs, info = env.reset()
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    for step in range(n_steps):
        # Get action
        action_mask = info['action_mask']
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
        
        # Track episode stats
        current_episode_reward += reward
        current_episode_length += 1
        
        if terminated or truncated:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            
            # Reset environment
            obs, info = env.reset()
        else:
            obs = next_obs
            info = next_info
            
    # Get last value for bootstrapping
    if not (terminated or truncated):
        action_mask = info['action_mask']
        _, pred_info = ppo.predict(obs, action_mask, deterministic=False)
        last_value = pred_info['value']
    else:
        last_value = 0.0
        
    # Compute returns and advantages
    buffer.compute_returns_and_advantages(
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        last_value=last_value
    )
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
        'n_episodes': len(episode_rewards)
    }


def train(
    snapshot_path: str,
    config: PPOConfig,
    checkpoint_dir: str = "checkpoints",
    tensorboard_dir: str = "tensorboard",
    save_freq: int = 10000,
    eval_freq: int = 5000
):
    """
    Train PPO model.
    
    Args:
        snapshot_path: Path to data snapshot
        config: PPO configuration
        checkpoint_dir: Directory for checkpoints
        tensorboard_dir: Directory for tensorboard logs
        save_freq: Save checkpoint frequency
        eval_freq: Evaluation frequency
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Create tensorboard writer
    run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(os.path.join(tensorboard_dir, run_name))
    
    # Create environment
    env = SchedulingEnv(snapshot_path)
    logger.info(f"Created environment with {env.n_tasks} tasks and {env.n_machines} machines")
    
    # Create PPO model
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        clip_range=config.clip_range,
        value_loss_coef=config.value_loss_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        gae_lambda=config.gae_lambda,
        gamma=config.gamma,
        device=config.device
    )
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=config.n_steps,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Training loop
    total_timesteps = 0
    n_updates = 0
    best_reward = -float('inf')
    
    pbar = tqdm(total=config.total_timesteps, desc="Training")
    
    while total_timesteps < config.total_timesteps:
        # Collect rollouts
        rollout_start = time.time()
        rollout_stats = collect_rollouts(env, ppo, buffer, config.n_steps)
        rollout_time = time.time() - rollout_start
        
        # Update model
        train_start = time.time()
        train_stats = ppo.train_on_buffer(buffer, writer)
        train_time = time.time() - train_start
        
        # Update counters
        total_timesteps += config.n_steps
        n_updates += 1
        ppo.n_episodes += rollout_stats['n_episodes']
        
        # Log stats
        writer.add_scalar('rollout/mean_reward', rollout_stats['mean_reward'], total_timesteps)
        writer.add_scalar('rollout/mean_length', rollout_stats['mean_length'], total_timesteps)
        writer.add_scalar('rollout/n_episodes', rollout_stats['n_episodes'], total_timesteps)
        writer.add_scalar('time/rollout', rollout_time, total_timesteps)
        writer.add_scalar('time/train', train_time, total_timesteps)
        
        # Update progress bar
        pbar.update(config.n_steps)
        pbar.set_postfix({
            'reward': f"{rollout_stats['mean_reward']:.2f}",
            'episodes': ppo.n_episodes,
            'updates': n_updates
        })
        
        # Save checkpoint
        if total_timesteps % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{total_timesteps}.pth"
            )
            ppo.save(checkpoint_path)
            
        # Save best model
        if rollout_stats['mean_reward'] > best_reward:
            best_reward = rollout_stats['mean_reward']
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            ppo.save(best_path)
            logger.info(f"New best model with reward {best_reward:.2f}")
            
        # Clear buffer
        buffer.reset()
        
    pbar.close()
    
    # Final save
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    ppo.save(final_path)
    
    # Close writer
    writer.close()
    
    logger.info(f"Training complete! Total timesteps: {total_timesteps}")
    logger.info(f"Best reward: {best_reward:.2f}")
    
    return ppo


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train PPO for scheduling")
    parser.add_argument(
        "--data",
        type=str,
        default="data/10_jobs.json",
        help="Path to data snapshot"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per rollout"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        total_timesteps=args.timesteps,
        device=args.device
    )
    
    # Train
    train(
        snapshot_path=args.data,
        config=config,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()