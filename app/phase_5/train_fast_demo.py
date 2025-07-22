#!/usr/bin/env python3
"""
Fast demo training for Phase 5 - Get results in reasonable time
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fast_demo")

def train_fast_demo():
    """Fast training with reduced parameters for quick results."""
    logger.info("="*60)
    logger.info("Phase 5 Fast Demo Training")
    logger.info("Target: Get results in <10 minutes")
    logger.info("="*60)
    
    # Fast demo configuration
    config = {
        'environment': {
            'n_machines': 145,
            'n_jobs': 100,  # Start with 100 jobs only
            'max_episode_steps': 500,
            'use_break_constraints': False,  # Disable for speed
            'use_holiday_constraints': False,
            'snapshot_file': 'data/real_production_snapshot.json'
        },
        'training': {
            'total_timesteps': 100000,  # Only 100k steps
            'n_envs': 4,  # Fewer environments for stability
            'learning_rate': 0.001,  # Higher LR for faster learning
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5
        }
    }
    
    # Create directories
    os.makedirs("models/multidiscrete/demo", exist_ok=True)
    os.makedirs("logs/phase5/demo", exist_ok=True)
    
    # Create environments
    logger.info(f"Creating {config['training']['n_envs']} training environments...")
    
    def make_env(rank):
        def _init():
            env = MultiDiscreteHierarchicalEnv(
                n_machines=config['environment']['n_machines'],
                n_jobs=config['environment']['n_jobs'],
                snapshot_file=config['environment']['snapshot_file'],
                max_episode_steps=config['environment']['max_episode_steps'],
                use_break_constraints=config['environment']['use_break_constraints'],
                use_holiday_constraints=config['environment']['use_holiday_constraints'],
                invalid_action_penalty=-10.0,  # Less harsh penalty
                seed=42 + rank
            )
            return Monitor(env, f"logs/phase5/demo/env_{rank}")
        return _init
    
    if config['training']['n_envs'] > 1:
        train_envs = SubprocVecEnv([make_env(i) for i in range(config['training']['n_envs'])])
    else:
        train_envs = DummyVecEnv([make_env(0)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(99)])
    
    # Create PPO model
    logger.info("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=config['training']['learning_rate'],
        n_steps=512,  # Smaller for faster updates
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_range=config['training']['clip_range'],
        ent_coef=config['training']['ent_coef'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/multidiscrete/demo/best",
        log_path="logs/phase5/demo/eval",
        eval_freq=10000 // config['training']['n_envs'],
        n_eval_episodes=3,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // config['training']['n_envs'],
        save_path="models/multidiscrete/demo/checkpoints",
        name_prefix="demo_model"
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train
    logger.info(f"\nStarting training for {config['training']['total_timesteps']:,} timesteps...")
    logger.info("This should take approximately 5-10 minutes\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nTraining time: {elapsed_time/60:.1f} minutes")
    
    # Save final model
    model.save("models/multidiscrete/demo/final_demo_model")
    logger.info("Model saved to: models/multidiscrete/demo/final_demo_model.zip")
    
    # Quick evaluation
    logger.info("\nEvaluating trained model...")
    obs = eval_env.reset()
    done = False
    episode_reward = 0
    steps = 0
    invalid_actions = 0
    scheduled_jobs = 0
    
    while not done and steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid_actions += 1
        else:
            scheduled_jobs = info[0].get('scheduled_count', 0)
    
    logger.info(f"\nDemo Training Results:")
    logger.info(f"  Episode reward: {episode_reward:.2f}")
    logger.info(f"  Steps taken: {steps}")
    logger.info(f"  Jobs scheduled: {scheduled_jobs}/{config['environment']['n_jobs']}")
    logger.info(f"  Invalid action rate: {invalid_actions/steps*100:.1f}%")
    
    if scheduled_jobs == config['environment']['n_jobs']:
        makespan = info[0].get('makespan', 0)
        logger.info(f"  Makespan: {makespan:.1f} hours")
        logger.info(f"  Success! All jobs scheduled.")
    else:
        logger.info(f"  Training needs more timesteps to complete all jobs.")
    
    logger.info("\n" + "="*60)
    logger.info("Demo complete! To run full training:")
    logger.info("  uv run python phase_5/train_multidiscrete_ppo.py")
    logger.info("="*60)

if __name__ == "__main__":
    train_fast_demo()