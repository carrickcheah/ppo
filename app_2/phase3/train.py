"""
Main Training Script for PPO Scheduling Model

Orchestrates the training process including:
- Environment setup
- Model initialization
- Rollout collection
- PPO updates
- Logging and checkpointing
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, List
import numpy as np
import torch
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.scheduling_game_env import SchedulingGameEnv
from src.data.data_loader import DataLoader
from phase2.ppo_scheduler import PPOScheduler
from phase2.rollout_buffer import RolloutBuffer
from phase2.curriculum import CurriculumManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Main trainer class for PPO scheduling model.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to training configuration YAML
        """
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load environment config
        env_config_path = os.path.join(
            os.path.dirname(config_path), 
            'environment.yaml'
        )
        with open(env_config_path, 'r') as f:
            self.env_config = yaml.safe_load(f)
            
        # Setup device
        device = self.config['training'].get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.curriculum_manager = CurriculumManager()
        self.model = PPOScheduler(self.config).to(self.device)
        
        # Training parameters
        self.n_envs = self.config['training'].get('n_envs', 8)
        self.n_steps = self.config['ppo'].get('n_steps', 2048)
        self.batch_size = self.config['ppo'].get('batch_size', 64)
        self.n_epochs = self.config['ppo'].get('n_epochs', 10)
        self.total_timesteps = self.config['training'].get('total_timesteps', 10_000_000)
        
        # Initialize environments
        self.envs = None
        self.n_jobs = None
        self.n_machines = None
        self._setup_environments()
        
        # Initialize rollout buffer
        obs_shape = self.envs[0].observation_space.shape
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_shape=obs_shape,
            action_shape=(),  # Scalar actions
            n_envs=self.n_envs,
            gamma=self.config['ppo'].get('gamma', 0.99),
            gae_lambda=self.config['ppo'].get('gae_lambda', 0.95),
            device=self.device
        )
        
        # Tracking
        self.timesteps = 0
        self.episodes = 0
        self.start_time = time.time()
        self.checkpoint_dir = self.config['tracking'].get('checkpoint_dir', './checkpoints/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
    def _setup_environments(self):
        """Setup training environments based on curriculum stage."""
        # Get curriculum configuration
        stage_config = self.curriculum_manager.get_env_config()
        
        # Update data loader config
        data_config = self.env_config['data'].copy()
        if stage_config.get('max_jobs'):
            data_config['max_jobs'] = stage_config['max_jobs']
        if stage_config.get('max_machines'):
            data_config['max_machines'] = stage_config['max_machines']
            
        # Create data loader
        data_loader = DataLoader(data_config)
        jobs = data_loader.load_jobs()
        machines = data_loader.load_machines()
        working_hours = data_loader.load_working_hours()
        
        # Store dimensions
        self.n_jobs = len(jobs)
        self.n_machines = len(machines)
        
        logger.info(f"Setting up {self.n_envs} environments with {self.n_jobs} jobs and {self.n_machines} machines")
        
        # Create environments
        self.envs = []
        for i in range(self.n_envs):
            env = SchedulingGameEnv(
                jobs=jobs.copy(),  # Copy to avoid shared state
                machines=machines.copy(),
                working_hours=working_hours,
                config=self.env_config
            )
            self.envs.append(env)
            
    def collect_rollouts(self) -> bool:
        """
        Collect rollouts from environments.
        
        Returns:
            True if any environment finished an episode
        """
        self.rollout_buffer.reset()
        
        # Get initial observations
        if not hasattr(self, 'last_obs'):
            self.last_obs = np.array([env.reset()[0] for env in self.envs])
            
        episode_finished = False
        
        for step in range(self.n_steps):
            # Get actions from policy
            actions = []
            values = []
            log_probs = []
            action_masks = []
            
            for i, env in enumerate(self.envs):
                obs = self.last_obs[i]
                
                # Get action mask
                mask = env.get_action_mask()
                action_masks.append(mask)
                
                # Get action from policy
                action, value, log_prob = self.model.get_action(
                    obs, mask, deterministic=False,
                    n_jobs=self.n_jobs, n_machines=self.n_machines
                )
                
                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)
                
            actions = np.array(actions)
            values = np.array(values)
            log_probs = np.array(log_probs)
            action_masks = np.array(action_masks)
            
            # Step environments
            new_obs = []
            rewards = []
            dones = []
            infos = []
            
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                # Decode action to (job, machine) format expected by env
                job_idx = action // self.n_machines
                machine_idx = action % self.n_machines
                env_action = np.array([job_idx, machine_idx])
                
                obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                
                new_obs.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                
                # Track episode statistics
                if done:
                    episode_finished = True
                    self.episodes += 1
                    
                    # Get episode stats from info
                    if 'episode_reward' in info:
                        self.episode_rewards.append(info['episode_reward'])
                    if 'episode_length' in info:
                        self.episode_lengths.append(info['episode_length'])
                        
                    # Reset environment
                    obs, _ = env.reset()
                    new_obs[i] = obs
                    
            new_obs = np.array(new_obs)
            rewards = np.array(rewards)
            dones = np.array(dones)
            
            # Add to buffer
            self.rollout_buffer.add(
                self.last_obs, actions, rewards, dones,
                values, log_probs, action_masks, infos
            )
            
            self.last_obs = new_obs
            self.timesteps += self.n_envs
            
        # Compute returns and advantages
        with torch.no_grad():
            # Get values for last states
            last_values = []
            for i, env in enumerate(self.envs):
                obs = self.last_obs[i]
                mask = env.get_action_mask()
                _, value, _ = self.model.get_action(
                    obs, mask, deterministic=True,
                    n_jobs=self.n_jobs, n_machines=self.n_machines
                )
                last_values.append(value)
            last_values = np.array(last_values)
            
        advantages, returns = self.rollout_buffer.compute_returns_and_advantages(last_values)
        self.rollout_buffer.add_computed_values(advantages, returns)
        
        return episode_finished
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total timesteps: {self.total_timesteps}")
        logger.info(f"Buffer size: {self.n_steps * self.n_envs}")
        
        while self.timesteps < self.total_timesteps:
            # Collect rollouts
            episode_finished = self.collect_rollouts()
            
            # Get data from buffer
            rollout_data = self.rollout_buffer.get()
            
            # Update policy
            update_metrics = self.model.update(
                rollout_data,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                n_jobs=self.n_jobs,
                n_machines=self.n_machines
            )
            
            # Update curriculum
            if episode_finished and self.episode_rewards:
                metrics = {
                    'episode_reward': np.mean(self.episode_rewards[-10:]),
                    'mean_reward': np.mean(self.episode_rewards[-10:])
                }
                
                advanced = self.curriculum_manager.update(
                    self.n_steps * self.n_envs, metrics
                )
                
                if advanced:
                    # Recreate environments for new stage
                    self._setup_environments()
                    self.last_obs = np.array([env.reset()[0] for env in self.envs])
                    
            # Log progress
            if self.timesteps % 10000 == 0:
                self._log_progress(update_metrics)
                
            # Save checkpoint
            if self.timesteps % self.config['training'].get('save_freq', 50000) == 0:
                self._save_checkpoint()
                
    def _log_progress(self, update_metrics: Dict[str, float]):
        """Log training progress."""
        elapsed_time = time.time() - self.start_time
        fps = self.timesteps / elapsed_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Timesteps: {self.timesteps}/{self.total_timesteps}")
        logger.info(f"Episodes: {self.episodes}")
        logger.info(f"FPS: {fps:.0f}")
        logger.info(f"Elapsed time: {elapsed_time/3600:.2f} hours")
        
        # Curriculum info
        curriculum_info = self.curriculum_manager.get_summary()
        logger.info(f"Curriculum stage: {curriculum_info['current_stage']}")
        logger.info(f"Stage progress: {curriculum_info['stage_progress']:.2%}")
        
        # Episode statistics
        if self.episode_rewards:
            logger.info(f"Mean episode reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
            logger.info(f"Std episode reward (last 100): {np.std(self.episode_rewards[-100:]):.2f}")
            
        # Update metrics
        for key, value in update_metrics.items():
            logger.info(f"{key}: {value:.4f}")
            
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.timesteps}.pt"
        )
        
        # Save model
        self.model.save(checkpoint_path)
        
        # Save curriculum progress
        curriculum_path = os.path.join(
            self.checkpoint_dir,
            f"curriculum_{self.timesteps}.pkl"
        )
        self.curriculum_manager.save_progress(curriculum_path)
        
        # Save training stats
        stats_path = os.path.join(
            self.checkpoint_dir,
            f"stats_{self.timesteps}.npz"
        )
        np.savez(
            stats_path,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            timesteps=self.timesteps,
            episodes=self.episodes
        )
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        # Load model
        self.model.load(checkpoint_path)
        
        # Extract timesteps from filename
        timesteps = int(os.path.basename(checkpoint_path).split('_')[1].split('.')[0])
        
        # Load curriculum progress
        curriculum_path = checkpoint_path.replace('checkpoint_', 'curriculum_').replace('.pt', '.pkl')
        if os.path.exists(curriculum_path):
            self.curriculum_manager.load_progress(curriculum_path)
            
        # Load training stats
        stats_path = checkpoint_path.replace('checkpoint_', 'stats_').replace('.pt', '.npz')
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.episode_rewards = list(stats['episode_rewards'])
            self.episode_lengths = list(stats['episode_lengths'])
            self.timesteps = int(stats['timesteps'])
            self.episodes = int(stats['episodes'])
            
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from timestep {self.timesteps}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO scheduling model')
    parser.add_argument('--config', type=str, default='configs/training.yaml',
                       help='Path to training configuration')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PPOTrainer(args.config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()