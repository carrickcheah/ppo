"""Training script for toy scheduling environment."""

import os
import sys
import argparse
from pathlib import Path
import yaml
from datetime import datetime

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environments.toy_env import ToySchedulingEnv


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(config: dict, seed: int = None) -> gym.Env:
    """Create and wrap the environment."""
    env_config = config['environment'].copy()
    env_config.pop('name', None)
    
    if seed is not None:
        env_config['seed'] = seed
    
    env = ToySchedulingEnv(**env_config)
    env = Monitor(env)
    return env


def train(config_path: str):
    """Train PPO on toy scheduling environment."""
    # Load configuration
    config = load_config(config_path)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup paths
    save_path = Path(config['training']['save_path']) / timestamp
    log_path = Path(config['training']['log_path']) / timestamp
    tensorboard_log = Path(config['training']['tensorboard_log'])
    
    # Create directories
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    
    # Save config to log directory
    with open(log_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Starting training run: {timestamp}")
    print(f"Saving models to: {save_path}")
    print(f"Logging to: {log_path}")
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = make_env(config, seed=config['seeds']['training'])
    train_env = DummyVecEnv([lambda: train_env])
    
    # Optionally normalize observations and rewards
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_env(config, seed=config['seeds']['eval'])
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / 'best_model'),
        log_path=str(log_path / 'eval'),
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(save_path / 'checkpoints'),
        name_prefix='ppo_toy_scheduler'
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Create PPO model
    print("\nCreating PPO model...")
    ppo_config = config['ppo'].copy()
    policy_kwargs = ppo_config.pop('policy_kwargs', {})
    
    # Convert activation function name to actual function
    if 'activation_fn' in policy_kwargs:
        import torch.nn as nn
        activation_name = policy_kwargs['activation_fn']
        if activation_name == 'tanh':
            policy_kwargs['activation_fn'] = nn.Tanh
        elif activation_name == 'relu':
            policy_kwargs['activation_fn'] = nn.ReLU
        # Add more activation functions as needed
    
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log),
        verbose=1,
        **ppo_config
    )
    
    # Train the model
    print(f"\nStarting training for {config['training']['total_timesteps']} timesteps...")
    print("You can monitor training progress with:")
    print(f"  tensorboard --logdir {tensorboard_log}")
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            tb_log_name=config['experiment']['name']
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_model_path = save_path / 'final_model'
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True
    )
    print(f"Final performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_env = make_env(config, seed=99)
    obs, _ = test_env.reset()
    
    print("\nRunning one episode with trained policy:")
    test_env.render()
    
    total_reward = 0
    for step in range(config['environment']['max_episode_steps']):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        
        if step % 5 == 0 or terminated or truncated:
            test_env.render()
        
        if terminated or truncated:
            print(f"\nEpisode finished after {step + 1} steps")
            print(f"Total reward: {total_reward:.2f}")
            if 'final_makespan' in info:
                print(f"Final makespan: {info['final_makespan']:.1f}")
                print(f"Machine utilization: {info['final_utilization']:.2%}")
                print(f"Jobs scheduled: {info['jobs_scheduled']}")
            break
    
    print("\nTraining complete!")
    return model, save_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train PPO on toy scheduling environment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/toy_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / args.config
        
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run training
    train(str(config_path))


if __name__ == '__main__':
    main()