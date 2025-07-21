#!/usr/bin/env python3
"""
Extended Phase 4 Training Script

This script extends the Phase 4 training from 1M to 2M timesteps with optimized
hyperparameters to achieve the target makespan of <45 hours.

Key optimizations:
- Increased learning rate (3e-5) for faster convergence
- Larger batch size (1024) for more stable updates
- Reduced entropy (0.005) for exploitation
- Learning rate scheduling for fine-tuning
- Checkpoint saving every 250k steps
"""

import os
import sys
import time
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from environments.full_production_env import FullProductionEnv
from data_ingestion.ingest_data import load_production_snapshot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtendedMetricsCallback(BaseCallback):
    """
    Custom callback to track makespan reduction over training.
    """
    
    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_makespan = float('inf')
        self.makespan_history = []
        self.step_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            obs = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.eval_env.step(action)
            
            makespan = info[0].get('makespan', float('inf'))
            self.makespan_history.append(makespan)
            self.step_history.append(self.num_timesteps)
            
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                logger.info(f"New best makespan: {makespan:.1f}h at step {self.num_timesteps}")
                
                # Save if we hit target
                if makespan < 45.0:
                    logger.info("TARGET ACHIEVED! Makespan < 45h")
                    self.model.save("models/full_production/target_achieved_model.zip")
            
            # Log progress
            if self.verbose:
                logger.info(f"Step {self.num_timesteps}: Makespan = {makespan:.1f}h")
                
        return True
    
    def _on_training_end(self) -> None:
        # Save metrics history
        metrics = {
            'makespan_history': self.makespan_history,
            'step_history': self.step_history,
            'best_makespan': self.best_makespan
        }
        
        with open('logs/phase4/extended_training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


class LearningRateSchedule:
    """
    Linear learning rate schedule that decays from initial to final value.
    """
    
    def __init__(self, initial_lr: float, final_lr: float, total_timesteps: int):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps
        
    def __call__(self, progress: float) -> float:
        """
        Progress is from 0 to 1 (percentage of training completed).
        """
        return self.final_lr + (self.initial_lr - self.final_lr) * (1 - progress)


def load_extended_config() -> Dict[str, Any]:
    """
    Load and update configuration for extended training.
    """
    # Load base config
    with open('configs/phase4_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update for extended training
    config['training']['total_timesteps'] = 2000000  # 2M steps
    config['training']['learning_rate'] = 0.00003  # 3e-5
    config['training']['batch_size'] = 1024
    config['training']['ent_coef'] = 0.005
    
    # Enable learning rate scheduling
    config['advanced_training']['use_lr_schedule'] = True
    config['advanced_training']['final_lr'] = 0.000001  # 1e-6
    
    # More frequent checkpoints
    config['logging']['save_freq'] = 250000  # Every 250k steps
    
    # Resume from best model
    config['training']['resume_from_checkpoint'] = True
    config['training']['checkpoint_path'] = "models/full_production/final_model.zip"
    
    return config


def create_env_fn(env_id: int, config: Dict[str, Any], snapshot_data: Dict[str, Any]):
    """
    Create environment function for vectorized environments.
    """
    def _init():
        env_config = {
            'n_machines': config['environment']['n_machines'],
            'use_break_constraints': config['environment']['use_break_constraints'],
            'use_holiday_constraints': config['environment']['use_holiday_constraints'],
            'state_compression': config['environment']['state_compression'],
            'seed': config['environment']['seed'] + env_id,
            'production_data': snapshot_data
        }
        
        env = FullProductionEnv(**env_config)
        return env
    
    set_random_seed(config['environment']['seed'] + env_id)
    return _init


def main():
    """
    Main training function for extended Phase 4.
    """
    start_time = time.time()
    logger.info("Starting Extended Phase 4 Training (2M timesteps)")
    
    # Load configuration
    config = load_extended_config()
    logger.info("Configuration loaded with optimized hyperparameters")
    
    # Create directories
    os.makedirs(config['models']['output_dir'], exist_ok=True)
    os.makedirs(config['models']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    
    # Load production data
    logger.info("Loading real production data...")
    snapshot_data = load_production_snapshot(config['environment']['snapshot_path'])
    logger.info(f"Loaded {len(snapshot_data['machines'])} machines, "
                f"{len(snapshot_data['jobs'])} jobs from snapshot")
    
    # Create vectorized environments
    logger.info(f"Creating {config['training']['n_envs']} parallel environments...")
    env_fns = [create_env_fn(i, config, snapshot_data) 
               for i in range(config['training']['n_envs'])]
    
    if config['training']['n_envs'] > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([create_env_fn(99, config, snapshot_data)])
    
    # Load existing model or create new one
    if config['training']['resume_from_checkpoint'] and \
       os.path.exists(config['training']['checkpoint_path']):
        logger.info(f"Loading checkpoint from {config['training']['checkpoint_path']}")
        model = PPO.load(
            config['training']['checkpoint_path'],
            env=train_env,
            device=config['training']['device']
        )
        
        # Update learning rate with schedule
        lr_schedule = LearningRateSchedule(
            config['training']['learning_rate'],
            config['advanced_training']['final_lr'],
            config['training']['total_timesteps']
        )
        model.learning_rate = lr_schedule
        
        # Update other hyperparameters
        model.batch_size = config['training']['batch_size']
        model.ent_coef = config['training']['ent_coef']
        
        logger.info("Model loaded and hyperparameters updated")
    else:
        raise ValueError("No checkpoint found for extended training!")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['save_freq'],
        save_path=config['models']['checkpoint_dir'],
        name_prefix="phase4_extended_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['models']['best_model_dir'],
        log_path=config['logging']['results_dir'],
        eval_freq=config['evaluation']['eval_freq'],
        n_eval_episodes=config['evaluation']['n_eval_episodes'],
        deterministic=config['evaluation']['deterministic']
    )
    
    metrics_callback = ExtendedMetricsCallback(
        eval_env=eval_env,
        eval_freq=config['evaluation']['eval_freq'],
        verbose=1
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback, metrics_callback])
    
    # Train the model
    logger.info("Starting extended training...")
    logger.info(f"Target: Reduce makespan from 49.2h to <45h")
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback_list,
            log_interval=config['logging']['log_interval'],
            tb_log_name="phase4_extended",
            reset_num_timesteps=False,  # Continue from existing timesteps
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(config['models']['output_dir'], 'extended_final_model.zip')
        model.save(final_path)
        logger.info(f"Extended training complete! Model saved to {final_path}")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        obs = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = eval_env.step(action)
        
        final_makespan = info[0].get('makespan', float('inf'))
        logger.info(f"Final makespan: {final_makespan:.1f}h")
        
        if final_makespan < 45.0:
            logger.info("SUCCESS! Target makespan achieved!")
        else:
            logger.info(f"Makespan improved but target not yet reached. "
                       f"Consider further hyperparameter tuning.")
        
        # Save training summary
        summary = {
            'training_time': time.time() - start_time,
            'total_timesteps': config['training']['total_timesteps'],
            'final_makespan': final_makespan,
            'best_makespan': metrics_callback.best_makespan,
            'initial_makespan': 49.2,
            'improvement': 49.2 - final_makespan,
            'target_achieved': final_makespan < 45.0,
            'config': config
        }
        
        with open(os.path.join(config['logging']['results_dir'], 
                              'extended_training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training completed in {(time.time() - start_time) / 3600:.1f} hours")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()