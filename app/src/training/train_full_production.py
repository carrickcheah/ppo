"""
Phase 4: Full Production Scale Training
Train PPO on 152 machines with 500+ jobs using curriculum learning approach.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import json
import yaml
import numpy as np
from typing import Dict, Any
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from ..environments.full_production_env import FullProductionEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetailedLoggingCallback(BaseCallback):
    """Custom callback for detailed logging during training."""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode results when available
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                
                if len(self.episode_rewards) % 100 == 0:
                    logger.info(f"Episodes: {len(self.episode_rewards)}, "
                              f"Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}, "
                              f"Avg Length: {np.mean(self.episode_lengths[-100:]):.1f}")
        return True


def load_config(config_path: str = "app/configs/phase4_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert activation function string to actual function
    activation_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU
    }
    
    activation_str = config['training'].get('activation_fn', 'tanh')
    config['training']['activation_fn'] = activation_map.get(activation_str, nn.Tanh)
    
    return config


def make_env(config: Dict[str, Any], rank: int, seed: int = 0):
    """Create a single environment instance."""
    def _init():
        env_config = config['environment']
        env = FullProductionEnv(
            n_machines=env_config['n_machines'],
            n_jobs=env_config['n_jobs'],
            state_compression=env_config['state_compression'],
            use_break_constraints=env_config['use_break_constraints'],
            use_holiday_constraints=env_config['use_holiday_constraints'],
            max_episode_steps=env_config['max_episode_steps'],
            seed=env_config.get('seed', 42) + rank
        )
        env = Monitor(env)
        return env
    return _init


def load_phase3_model(config: Dict[str, Any]):
    """Load the trained Phase 3 model."""
    phase3_path = config['models']['phase3_model_path']
    if not Path(phase3_path).exists():
        logger.warning(f"Phase 3 model not found at {phase3_path}")
        logger.info("Will train from scratch instead")
        return None
        
    logger.info(f"Loading Phase 3 model from {phase3_path}")
    
    try:
        model = PPO.load(
            phase3_path,
            device=config['training'].get('device', 'auto')
        )
        logger.info("Successfully loaded Phase 3 model")
        return model
    except Exception as e:
        logger.error(f"Error loading Phase 3 model: {e}")
        return None


def evaluate_model(model, config: Dict[str, Any], n_eval_episodes: int = 5) -> Dict[str, Any]:
    """Evaluate model performance."""
    logger.info(f"Evaluating model for {n_eval_episodes} episodes...")
    
    # Create evaluation environment
    env_config = config['environment']
    eval_env = FullProductionEnv(
        n_machines=env_config['n_machines'],
        n_jobs=env_config['n_jobs'],
        state_compression=env_config['state_compression'],
        use_break_constraints=env_config['use_break_constraints'],
        use_holiday_constraints=env_config['use_holiday_constraints'],
        seed=9999
    )
    
    results = []
    
    for episode in range(n_eval_episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            steps += 1
            
        stats = eval_env.get_stats()
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': steps,
            'makespan': stats['makespan'],
            'completion_rate': stats['completion_rate'],
            'avg_utilization': stats['avg_utilization'],
            'jobs_scheduled': stats['n_jobs_scheduled'],
            'total_jobs': len(eval_env.jobs)
        })
        
    eval_env.close()
    
    # Aggregate results
    avg_results = {
        'avg_reward': np.mean([r['reward'] for r in results]),
        'avg_makespan': np.mean([r['makespan'] for r in results]),
        'avg_completion_rate': np.mean([r['completion_rate'] for r in results]),
        'avg_utilization': np.mean([r['avg_utilization'] for r in results]),
        'all_results': results
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Average reward: {avg_results['avg_reward']:.2f}")
    logger.info(f"  Average makespan: {avg_results['avg_makespan']:.2f}h")
    logger.info(f"  Average completion rate: {avg_results['avg_completion_rate']:.2%}")
    logger.info(f"  Average utilization: {avg_results['avg_utilization']:.2%}")
    
    return avg_results


def train_phase4(config_path: str = "app/configs/phase4_config.yaml"):
    """Main training function for Phase 4."""
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(config['models']['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Phase 4: Full Production Scale Training ===")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Machines: {config['environment']['n_machines']}")
    logger.info(f"Jobs: {config['environment']['n_jobs']}")
    logger.info(f"State compression: {config['environment']['state_compression']}")
    
    # Create vectorized environment
    n_envs = config['training']['n_envs']
    logger.info(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(config, 9999)])
    
    # Load Phase 3 model or create new one
    phase3_model = load_phase3_model(config)
    
    if phase3_model is not None:
        # Transfer learning from Phase 3
        logger.info("Initializing model with Phase 3 weights...")
        
        # Create new model with same architecture but new environment
        train_config = config['training']
        policy_kwargs = dict(
            net_arch=train_config['network_arch'],
            activation_fn=train_config['activation_fn']
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_config['learning_rate'],
            n_steps=train_config['n_steps'],
            batch_size=train_config['batch_size'],
            n_epochs=train_config['n_epochs'],
            gamma=train_config['gamma'],
            clip_range=train_config['clip_range'],
            ent_coef=train_config['ent_coef'],
            vf_coef=train_config.get('vf_coef', 0.5),
            max_grad_norm=train_config.get('max_grad_norm', 0.5),
            gae_lambda=train_config.get('gae_lambda', 0.95),
            policy_kwargs=policy_kwargs,
            verbose=train_config.get('verbose', 1),
            device=train_config.get('device', 'auto')
        )
        
        # Transfer weights (if compatible)
        try:
            # Get state dicts
            phase3_state = phase3_model.policy.state_dict()
            phase4_state = model.policy.state_dict()
            
            # Transfer compatible layers
            transferred = 0
            for key in phase4_state.keys():
                if key in phase3_state and phase3_state[key].shape == phase4_state[key].shape:
                    phase4_state[key] = phase3_state[key]
                    transferred += 1
                    
            model.policy.load_state_dict(phase4_state)
            logger.info(f"Transferred {transferred}/{len(phase4_state)} layers from Phase 3")
            
        except Exception as e:
            logger.warning(f"Could not transfer all weights: {e}")
            logger.info("Will fine-tune from partially transferred weights")
            
        # Evaluate initial performance
        logger.info("\nEvaluating Phase 3 model on full production scale...")
        initial_results = evaluate_model(phase3_model, config, n_eval_episodes=3)
        
    else:
        # Train from scratch
        logger.info("Creating new model (no Phase 3 model found)...")
        train_config = config['training']
        policy_kwargs = dict(
            net_arch=train_config['network_arch'],
            activation_fn=train_config['activation_fn']
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_config['learning_rate'],
            n_steps=train_config['n_steps'],
            batch_size=train_config['batch_size'],
            n_epochs=train_config['n_epochs'],
            gamma=train_config['gamma'],
            clip_range=train_config['clip_range'],
            ent_coef=train_config['ent_coef'],
            vf_coef=train_config.get('vf_coef', 0.5),
            max_grad_norm=train_config.get('max_grad_norm', 0.5),
            gae_lambda=train_config.get('gae_lambda', 0.95),
            policy_kwargs=policy_kwargs,
            verbose=train_config.get('verbose', 1),
            device=train_config.get('device', 'auto')
        )
        initial_results = None
        
    # Set up callbacks
    eval_config = config['evaluation']
    log_config = config['logging']
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['models']['best_model_dir'],
        log_path=str(output_path / "eval_logs"),
        eval_freq=eval_config['eval_freq'],
        deterministic=eval_config['deterministic'],
        render=eval_config['render'],
        n_eval_episodes=eval_config['n_eval_episodes']
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=log_config['save_freq'],
        save_path=config['models']['checkpoint_dir'],
        name_prefix="phase4_model"
    )
    
    logging_callback = DetailedLoggingCallback(
        log_dir=str(output_path / "detailed_logs")
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback, logging_callback])
    
    # Train the model
    total_timesteps = config['training']['total_timesteps']
    logger.info(f"\nStarting training for {total_timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save final model
    final_model_path = output_path / "final_model"
    model.save(str(final_model_path))
    logger.info(f"Saved final model to {final_model_path}")
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    final_results = evaluate_model(model, config, n_eval_episodes=5)
    
    # Save results
    results_data = {
        "config": config,
        "initial_results": initial_results,
        "final_results": final_results,
        "training_time_seconds": training_time,
        "phase3_model_used": phase3_model is not None
    }
    
    with open(output_path / "training_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
        
    # Compare with baselines
    logger.info("\nComparing with baseline policies...")
    baseline_results = evaluate_baselines(config)
    
    # Summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Scale: {config['environment']['n_machines']} machines, {config['environment']['n_jobs']} jobs")
    logger.info(f"Training time: {training_time/60:.1f} minutes")
    
    if initial_results:
        logger.info(f"\nPhase 3 model performance:")
        logger.info(f"  Makespan: {initial_results['avg_makespan']:.2f}h")
        
    logger.info(f"\nPhase 4 model performance:")
    logger.info(f"  Makespan: {final_results['avg_makespan']:.2f}h")
    logger.info(f"  Completion rate: {final_results['avg_completion_rate']:.2%}")
    
    logger.info(f"\nBaseline comparison:")
    for name, result in baseline_results.items():
        logger.info(f"  {name}: {result['avg_makespan']:.2f}h")
        
    # Cleanup
    env.close()
    eval_env.close()
    
    return model, results_data


def evaluate_baselines(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Evaluate baseline policies for comparison."""
    from ..environments.baselines import RandomPolicy, FirstFitPolicy
    
    baselines = {
        'Random': RandomPolicy(),
        'FirstFit': FirstFitPolicy()
    }
    
    results = {}
    
    for name, policy in baselines.items():
        logger.info(f"\nEvaluating {name} baseline...")
        
        env_config = config['environment']
        env = FullProductionEnv(
            n_machines=env_config['n_machines'],
            n_jobs=env_config['n_jobs'],
            state_compression=env_config['state_compression'],
            use_break_constraints=env_config['use_break_constraints'],
            use_holiday_constraints=env_config['use_holiday_constraints'],
            seed=9999
        )
        
        episode_results = []
        
        for episode in range(3):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            
            while not (terminated or truncated):
                action = policy.get_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
            stats = env.get_stats()
            episode_results.append({
                'reward': episode_reward,
                'makespan': stats['makespan'],
                'completion_rate': stats['completion_rate']
            })
            
        env.close()
        
        results[name] = {
            'avg_makespan': np.mean([r['makespan'] for r in episode_results]),
            'avg_completion_rate': np.mean([r['completion_rate'] for r in episode_results])
        }
        
    return results


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "app/configs/phase4_config.yaml"
    train_phase4(config_path)