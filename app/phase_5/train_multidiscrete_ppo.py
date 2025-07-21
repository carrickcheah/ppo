#!/usr/bin/env python3
"""
Train PPO with MultiDiscrete Action Space for Phase 5

This script trains a PPO model using the MultiDiscrete hierarchical
environment, which is compatible with Stable Baselines3.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import yaml
import logging
from datetime import datetime
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_multidiscrete")


def load_config(config_path: str = "configs/phase5_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, rank: int = 0, eval_env: bool = False) -> MultiDiscreteHierarchicalEnv:
    """
    Create a MultiDiscrete hierarchical production environment.
    """
    env_config = config['environment']
    hier_config = config.get('hierarchical', {})
    
    # For curriculum learning, start with fewer jobs
    if not eval_env and config['training'].get('curriculum', {}).get('enabled', False):
        # Start with first curriculum stage
        n_jobs = config['training']['curriculum']['stages'][0]
        logger.info(f"Curriculum learning: Starting with {n_jobs} jobs")
    else:
        n_jobs = env_config['n_jobs']
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=env_config['n_machines'],
        n_jobs=n_jobs,
        data_file=env_config.get('data_path'),
        snapshot_file=env_config.get('snapshot_file'),
        max_episode_steps=env_config['max_episode_steps'],
        use_break_constraints=env_config['use_break_constraints'],
        use_holiday_constraints=env_config['use_holiday_constraints'],
        invalid_action_penalty=hier_config.get('invalid_action_penalty', -20.0),
        seed=42 + rank if not eval_env else 123
    )
    
    # Wrap in monitor
    log_dir = f"logs/phase5/env_{rank}" if not eval_env else "logs/phase5/eval"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    return env


def make_env(config: dict, rank: int):
    """Create environment function for vectorization."""
    def _init():
        return create_env(config, rank=rank)
    return _init


class CurriculumCallback:
    """Callback to implement curriculum learning."""
    
    def __init__(self, config: dict, env_creator):
        self.config = config
        self.env_creator = env_creator
        self.curriculum = config['training'].get('curriculum', {})
        self.stages = self.curriculum.get('stages', [100])
        self.stage_timesteps = self.curriculum.get('stage_timesteps', 500000)
        self.current_stage = 0
        
    def __call__(self, locals_, globals_):
        # Check if we should advance to next stage
        timesteps = locals_['self'].num_timesteps
        target_stage = min(timesteps // self.stage_timesteps, len(self.stages) - 1)
        
        if target_stage > self.current_stage:
            self.current_stage = target_stage
            n_jobs = self.stages[self.current_stage]
            
            logger.info(f"\nAdvancing curriculum to stage {self.current_stage + 1}: {n_jobs} jobs")
            
            # Create new environments with more jobs
            # Note: This is a simplified version. In practice, you'd need to
            # properly update the vectorized environment
        
        return True


def train_multidiscrete_ppo():
    """
    Main training function for MultiDiscrete hierarchical PPO.
    """
    logger.info("="*60)
    logger.info("Starting MultiDiscrete Hierarchical PPO Training")
    logger.info("Phase 5: Solving Action Space Limitation")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Extract parameters
    train_config = config['training']
    env_config = config['environment']
    
    # Create directories
    os.makedirs("models/multidiscrete", exist_ok=True)
    os.makedirs("logs/phase5", exist_ok=True)
    os.makedirs("logs/phase5/tensorboard", exist_ok=True)
    os.makedirs("visualizations/phase_5", exist_ok=True)
    
    # Create training environments
    n_envs = train_config.get('n_envs', 4)
    logger.info(f"Creating {n_envs} training environments...")
    
    if n_envs > 1:
        # Use subprocess for true parallelism
        train_envs = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    else:
        # Single environment
        train_envs = DummyVecEnv([make_env(config, 0)])
    
    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: create_env(config, eval_env=True)])
    
    # Set up tensorboard logging
    tb_log_name = f"multidiscrete_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create PPO model
    logger.info("\nCreating PPO model...")
    logger.info(f"  Action space: MultiDiscrete")
    logger.info(f"  Learning rate: {train_config['learning_rate']}")
    logger.info(f"  Batch size: {train_config['batch_size']}")
    logger.info(f"  N epochs: {train_config['n_epochs']}")
    
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=train_config['learning_rate'],
        n_steps=2048 // n_envs,  # Adjust for number of environments
        batch_size=train_config['batch_size'],
        n_epochs=train_config['n_epochs'],
        gamma=train_config['gamma'],
        gae_lambda=train_config['gae_lambda'],
        clip_range=train_config['clip_range'],
        clip_range_vf=None,
        ent_coef=train_config['ent_coef'],
        vf_coef=train_config['vf_coef'],
        max_grad_norm=0.5,
        target_kl=None,
        tensorboard_log=f"logs/phase5/tensorboard/{tb_log_name}",
        verbose=1
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config.get('checkpoint_freq', 100000) // n_envs, 1),
        save_path="models/multidiscrete/checkpoints",
        name_prefix="multidiscrete_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/multidiscrete/best",
        log_path="logs/phase5/eval",
        eval_freq=max(train_config.get('eval_freq', 50000) // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Log training info
    logger.info("\nTraining configuration:")
    logger.info(f"  Total timesteps: {train_config['total_timesteps']:,}")
    logger.info(f"  Environments: {n_envs}")
    logger.info(f"  Checkpoint frequency: {train_config.get('checkpoint_freq', 100000):,}")
    logger.info(f"  Evaluation frequency: {train_config.get('eval_freq', 50000):,}")
    
    if train_config.get('curriculum', {}).get('enabled', False):
        stages = train_config['curriculum']['stages']
        logger.info(f"\nCurriculum learning enabled:")
        logger.info(f"  Stages: {stages}")
        logger.info(f"  Timesteps per stage: {train_config['curriculum']['stage_timesteps']:,}")
    
    # Train the model
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=train_config['total_timesteps'],
            callback=callback_list,
            progress_bar=True,
            tb_log_name=tb_log_name
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining error: {e}")
        raise
    
    # Training completed
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
    logger.info("="*60)
    
    # Save final model
    final_path = "models/multidiscrete/final_model"
    model.save(final_path)
    logger.info(f"\nFinal model saved to: {final_path}")
    
    # Test the final model
    logger.info("\nTesting final model...")
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    steps = 0
    invalid_actions = 0
    
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid_actions += 1
    
    logger.info(f"\nTest episode results:")
    logger.info(f"  Total reward: {episode_reward:.2f}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Invalid actions: {invalid_actions} ({100*invalid_actions/steps:.1f}%)")
    logger.info(f"  Jobs scheduled: {info[0].get('scheduled_count', 0)}")
    
    logger.info("\n" + "="*60)
    logger.info("Phase 5 MultiDiscrete PPO training complete!")
    logger.info("Next steps:")
    logger.info("  1. Evaluate model performance")
    logger.info("  2. Compare with Phase 4 batch processing")
    logger.info("  3. Measure makespan improvement")
    logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MultiDiscrete PPO for Phase 5")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/phase5_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run short demo training"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Override config for demo
        logger.info("Running demo mode with reduced timesteps...")
        # TODO: Implement demo mode
    
    train_multidiscrete_ppo()