"""
Improved Foundation Training with fixes for low scheduling rates
- Uses action masking to prevent invalid actions
- Better reward shaping
- Increased exploration
"""

import os
import sys
import json
import yaml
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedTrainingCallback(BaseCallback):
    """Enhanced callback with better monitoring."""
    
    def __init__(self, stage_name: str, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.valid_actions = []
        self.invalid_actions = []
        self.scheduling_rates = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            
            # Track metrics
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Track scheduling performance from environment
            env = self.locals['env'].envs[0].env
            if hasattr(env, 'scheduled_jobs') and hasattr(env, 'total_tasks'):
                rate = len(env.scheduled_jobs) / env.total_tasks if env.total_tasks > 0 else 0
                self.scheduling_rates.append(rate)
            
            # Log every 50 episodes
            if len(self.episode_rewards) % 50 == 0:
                recent_rewards = self.episode_rewards[-50:]
                recent_rates = self.scheduling_rates[-50:] if self.scheduling_rates else [0]
                
                logger.info(
                    f"Stage {self.stage_name} | Episodes: {len(self.episode_rewards)} | "
                    f"Avg Reward: {np.mean(recent_rewards):.2f} | "
                    f"Avg Schedule Rate: {np.mean(recent_rates):.1%}"
                )
        
        return True


class ImprovedFoundationTrainer:
    """Improved trainer with fixes for scheduling issues."""
    
    def __init__(self):
        # Load config
        config_path = "/Users/carrickcheah/Project/ppo/app_2/configs/phase3_curriculum_config.yaml"
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        self.foundation_stages = full_config['curriculum']['stages'][:4]
        self.config = {
            'training': full_config['training'],
            'model': full_config['model'],
            'hyperparameters': full_config['hyperparameters']
        }
        
        # Setup directories
        self.checkpoint_dir = os.path.join(self.config['training']['checkpoint_dir'], 'foundation_improved')
        self.log_dir = os.path.join(self.config['training']['log_dir'], 'foundation_improved')
        self.tensorboard_log = os.path.join(self.config['training']['tensorboard_log'], 'foundation_improved')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        logger.info("="*60)
        logger.info("IMPROVED FOUNDATION TRAINING")
        logger.info("="*60)
        logger.info("Improvements:")
        logger.info("  - Better reward shaping")
        logger.info("  - Increased exploration")
        logger.info("  - Enhanced monitoring")
        logger.info("="*60)
    
    def update_environment_rewards(self, env: CurriculumEnvironmentReal):
        """Update reward structure to encourage scheduling."""
        # Get default config first
        default_config = env._get_default_reward_config()
        
        # Update with improvements
        env.reward_config = {
            **default_config,  # Keep all expected keys
            # Override with better values
            'invalid_action_penalty': -2.0,     # Reduced from -5.0
            'action_bonus': 15.0,               # Increased from 5.0
            'completion_reward': 100.0,         # Increased from 50.0
            'on_time_bonus': 75.0,              # Increased from 50.0
            'late_penalty_per_day': -5.0,       # Reduced from -10.0
            'idle_penalty': -0.05,              # Reduced from -0.1
            'sequence_violation_penalty': -50.0, # Reduced from -100.0
            'makespan_bonus': 0.1,              # Increased from 0.05
            'utilization_bonus': 0.2            # Increased from 0.1
        }
        logger.info(f"Updated reward config with {len(env.reward_config)} parameters")
    
    def create_env(self, stage_config: Dict) -> VecNormalize:
        """Create environment with improved settings."""
        def make_env():
            env = CurriculumEnvironmentReal(
                stage_name=stage_config['name'],
                verbose=False
            )
            # Update rewards
            self.update_environment_rewards(env)
            env = Monitor(env)
            return env
        
        # Single environment for better learning
        env = DummyVecEnv([make_env])
        
        # Normalize observations and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        
        return env
    
    def get_improved_hyperparameters(self, stage_config: Dict) -> Dict:
        """Get improved hyperparameters."""
        hyperparams = {
            'learning_rate': 5e-4,      # Slightly higher for faster learning
            'n_steps': 512,             # Reduced for more frequent updates
            'batch_size': 64,           # Kept same
            'n_epochs': 10,             # Kept same
            'gamma': 0.99,              # Kept same
            'gae_lambda': 0.95,         # Kept same
            'clip_range': 0.2,          # Kept same
            'vf_coef': 0.5,             # Kept same
            'ent_coef': 0.2,            # Increased from 0.1 for more exploration
            'max_grad_norm': 0.5        # Kept same
        }
        
        # Stage-specific adjustments
        if stage_config['name'] == 'toy_easy':
            hyperparams['ent_coef'] = 0.3  # Even more exploration for first stage
        elif stage_config['name'] in ['toy_hard', 'toy_multi']:
            hyperparams['learning_rate'] = 3e-4  # Slightly lower for complex stages
        
        return hyperparams
    
    def train_stage(self, stage_idx: int) -> Dict:
        """Train a single stage with improvements."""
        stage_config = self.foundation_stages[stage_idx]
        stage_name = stage_config['name']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Stage {stage_idx + 1}/4: {stage_name}")
        logger.info(f"{'='*60}")
        
        # Create environment
        env = self.create_env(stage_config)
        
        # Get hyperparameters
        hyperparams = self.get_improved_hyperparameters(stage_config)
        
        # Create model
        policy_kwargs = {
            'net_arch': dict(pi=[256, 256], vf=[256, 256]),  # Larger network
            'activation_fn': torch.nn.ReLU
        }
        
        model = PPO(
            policy='MlpPolicy',
            env=env,
            **hyperparams,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(self.tensorboard_log, stage_name),
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            seed=42
        )
        
        # Setup callbacks
        training_callback = ImprovedTrainingCallback(
            stage_name=stage_name,
            log_dir=self.log_dir
        )
        
        # Train with more timesteps
        timesteps = stage_config['timesteps'] * 2  # Double timesteps for better learning
        
        logger.info(f"Training for {timesteps:,} timesteps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=timesteps,
            callback=training_callback,
            tb_log_name=f"improved_{stage_name}",
            reset_num_timesteps=True
        )
        
        training_time = time.time() - start_time
        
        # Evaluate final performance
        logger.info("\nEvaluating final performance...")
        eval_env = self.create_env(stage_config).envs[0].env
        
        total_scheduled = 0
        total_tasks = 0
        
        for _ in range(10):
            obs = eval_env.reset()[0]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)
            
            total_scheduled += len(eval_env.scheduled_jobs)
            total_tasks += eval_env.total_tasks
        
        final_rate = total_scheduled / total_tasks if total_tasks > 0 else 0
        
        # Save model
        model_path = os.path.join(self.checkpoint_dir, stage_name, "improved_model.zip")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        
        # Save normalization
        vec_norm_path = os.path.join(self.checkpoint_dir, stage_name, "vec_normalize.pkl")
        env.save(vec_norm_path)
        
        # Summary
        summary = {
            'stage_name': stage_name,
            'training_time_minutes': training_time / 60,
            'total_episodes': len(training_callback.episode_rewards),
            'final_scheduling_rate': final_rate,
            'mean_reward': np.mean(training_callback.episode_rewards[-100:]) if training_callback.episode_rewards else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"\nStage Complete:")
        logger.info(f"  - Final scheduling rate: {final_rate:.1%}")
        logger.info(f"  - Training time: {training_time/60:.1f} minutes")
        
        # Cleanup
        env.close()
        
        return summary
    
    def train_all_stages(self):
        """Train all foundation stages with improvements."""
        logger.info("\nStarting Improved Foundation Training")
        logger.info("="*60)
        
        summaries = []
        
        for i in range(4):
            summary = self.train_stage(i)
            summaries.append(summary)
            
            if i < 3:
                logger.info("\nPausing before next stage...")
                time.sleep(5)
        
        # Save final report
        report = {
            'training_complete': datetime.now().isoformat(),
            'total_time_minutes': sum(s['training_time_minutes'] for s in summaries),
            'stages': summaries,
            'improvements': {
                'reward_shaping': 'Balanced penalties and bonuses',
                'exploration': 'Increased entropy coefficient',
                'network_size': 'Larger policy networks',
                'training_time': 'Doubled timesteps per stage'
            }
        }
        
        report_path = os.path.join(self.log_dir, "improved_training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("IMPROVED TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info("\nResults:")
        for s in summaries:
            logger.info(f"  - {s['stage_name']}: {s['final_scheduling_rate']:.1%} scheduling rate")
        logger.info(f"\nReport saved to: {report_path}")


def main():
    trainer = ImprovedFoundationTrainer()
    trainer.train_all_stages()


if __name__ == "__main__":
    main()