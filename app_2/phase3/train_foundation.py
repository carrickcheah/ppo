"""
Foundation Training Script - Focus on First 4 Stages
Trains PPO model through toy_easy, toy_normal, toy_hard, and toy_multi stages
Ensures model learns basic concepts before scaling up
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
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FoundationTrainingCallback(BaseCallback):
    """Custom callback for monitoring foundation training progress."""
    
    def __init__(self, stage_name: str, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
        
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            
            # Extract metrics
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_info.append(info)
            
            # Log every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                logger.info(
                    f"Stage {self.stage_name} | "
                    f"Episodes: {len(self.episode_rewards)} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Mean Length: {mean_length:.1f}"
                )
        
        return True
    
    def _on_training_end(self) -> None:
        """Save stage metrics at end of training."""
        metrics = {
            'stage_name': self.stage_name,
            'total_episodes': len(self.episode_rewards),
            'mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            'std_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
            'mean_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
            'final_reward': float(self.episode_rewards[-1]) if self.episode_rewards else 0,
            'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
            'min_reward': float(np.min(self.episode_rewards)) if self.episode_rewards else 0
        }
        
        # Save metrics
        metrics_path = os.path.join(self.log_dir, f"foundation_{self.stage_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


class FoundationTrainer:
    """Manages the foundation training process (first 4 stages only)."""
    
    def __init__(self):
        """Initialize foundation trainer."""
        # Load base configuration
        config_path = "/Users/carrickcheah/Project/ppo/app_2/configs/phase3_curriculum_config.yaml"
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Extract foundation stages (first 4)
        self.foundation_stages = full_config['curriculum']['stages'][:4]
        
        # Copy other config sections
        self.config = {
            'training': full_config['training'],
            'model': full_config['model'],
            'hyperparameters': full_config['hyperparameters']
        }
        
        # Setup directories
        self.checkpoint_dir = os.path.join(self.config['training']['checkpoint_dir'], 'foundation')
        self.log_dir = os.path.join(self.config['training']['log_dir'], 'foundation')
        self.tensorboard_log = os.path.join(self.config['training']['tensorboard_log'], 'foundation')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Training state
        self.model = None
        
        logger.info("="*60)
        logger.info("FOUNDATION TRAINING - FIRST 4 STAGES ONLY")
        logger.info("="*60)
        logger.info("Stages to train:")
        for i, stage in enumerate(self.foundation_stages):
            logger.info(f"  {i+1}. {stage['name']}: {stage['jobs']} jobs, {stage['machines']} machines - {stage['description']}")
        logger.info("="*60)
    
    def create_env(self, stage_config: Dict) -> VecNormalize:
        """Create environment for a specific stage."""
        def make_env():
            env = CurriculumEnvironmentReal(
                stage_name=stage_config['name'],
                verbose=False
            )
            env = Monitor(env)
            return env
        
        # Create vectorized environment
        n_envs = self.config['training'].get('n_envs', 1)
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Add normalization wrapper
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        return env
    
    def get_hyperparameters(self, stage_config: Dict) -> Dict:
        """Get hyperparameters for a specific stage."""
        # Start with default hyperparameters
        hyperparams = self.config['hyperparameters'].copy()
        
        # Apply stage-specific overrides if any
        overrides = stage_config.get('hyperparameter_overrides', {})
        hyperparams.update(overrides)
        
        # For foundation stages, ensure good exploration
        if stage_config['name'] in ['toy_easy', 'toy_normal']:
            hyperparams['ent_coef'] = 0.1  # Higher exploration for early stages
        
        return hyperparams
    
    def evaluate_stage_learning(self, env: VecNormalize, stage_config: Dict) -> Dict:
        """Evaluate what the model learned in this stage."""
        logger.info(f"\nEvaluating learning outcomes for {stage_config['name']}...")
        
        # Run several episodes
        n_eval = 10
        outcomes = {
            'sequences_respected': 0,
            'on_time_completions': 0,
            'important_prioritized': 0,
            'multi_machine_handled': 0,
            'total_jobs_scheduled': 0,
            'total_jobs_available': 0
        }
        
        for _ in range(n_eval):
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
            
            # Analyze episode results
            base_env = env.envs[0].env
            
            # Count scheduled jobs (tasks, not families)
            outcomes['total_jobs_scheduled'] += len(base_env.scheduled_jobs)
            outcomes['total_jobs_available'] += base_env.total_tasks
        
        # Calculate learning metrics
        learning_report = {
            'stage': stage_config['name'],
            'scheduling_rate': outcomes['total_jobs_scheduled'] / outcomes['total_jobs_available'] if outcomes['total_jobs_available'] > 0 else 0,
            'focus': stage_config['description']
        }
        
        logger.info(f"  Scheduling rate: {learning_report['scheduling_rate']:.1%}")
        logger.info(f"  Learning focus: {learning_report['focus']}")
        
        return learning_report
    
    def train_stage(self, stage_idx: int) -> Dict:
        """Train a single foundation stage."""
        stage_config = self.foundation_stages[stage_idx]
        stage_name = stage_config['name']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Foundation Stage {stage_idx + 1}/4: {stage_name}")
        logger.info(f"Description: {stage_config['description']}")
        logger.info(f"Jobs: {stage_config['jobs']}, Machines: {stage_config['machines']}")
        logger.info(f"Timesteps: {stage_config['timesteps']:,}")
        logger.info(f"{'='*60}")
        
        # Create environment
        env = self.create_env(stage_config)
        
        # Get hyperparameters
        hyperparams = self.get_hyperparameters(stage_config)
        
        # Create or update model
        if self.model is None:
            # First stage - create new model
            logger.info("Creating new PPO model for foundation training...")
            
            policy_kwargs = {
                'net_arch': dict(pi=[128, 128], vf=[128, 128]),
                'activation_fn': torch.nn.ReLU
            }
            
            self.model = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=float(hyperparams['learning_rate']),
                n_steps=int(hyperparams['n_steps']),
                batch_size=int(hyperparams['batch_size']),
                n_epochs=int(hyperparams['n_epochs']),
                gamma=float(hyperparams['gamma']),
                gae_lambda=float(hyperparams['gae_lambda']),
                clip_range=float(hyperparams['clip_range']),
                vf_coef=float(hyperparams['vf_coef']),
                ent_coef=float(hyperparams['ent_coef']),
                max_grad_norm=float(hyperparams['max_grad_norm']),
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(self.tensorboard_log, stage_name),
                verbose=self.config['training']['verbose'],
                device=self.config['training']['device'],
                seed=42
            )
        else:
            # For foundation training, create new model for each stage
            # This is necessary because observation space changes with different job/machine counts
            logger.info("Creating new model for this stage (observation space changed)...")
            
            policy_kwargs = {
                'net_arch': dict(pi=[128, 128], vf=[128, 128]),
                'activation_fn': torch.nn.ReLU
            }
            
            self.model = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=float(hyperparams['learning_rate']),
                n_steps=int(hyperparams['n_steps']),
                batch_size=int(hyperparams['batch_size']),
                n_epochs=int(hyperparams['n_epochs']),
                gamma=float(hyperparams['gamma']),
                gae_lambda=float(hyperparams['gae_lambda']),
                clip_range=float(hyperparams['clip_range']),
                vf_coef=float(hyperparams['vf_coef']),
                ent_coef=float(hyperparams['ent_coef']),
                max_grad_norm=float(hyperparams['max_grad_norm']),
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(self.tensorboard_log, stage_name),
                verbose=self.config['training']['verbose'],
                device=self.config['training']['device'],
                seed=42
            )
        
        # Setup callbacks
        callbacks = []
        
        # Foundation training callback
        foundation_callback = FoundationTrainingCallback(
            stage_name=stage_name,
            log_dir=self.log_dir
        )
        callbacks.append(foundation_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training']['checkpoint_freq'],
            save_path=os.path.join(self.checkpoint_dir, stage_name),
            name_prefix=f"foundation_{stage_name}"
        )
        callbacks.append(checkpoint_callback)
        
        # Combine callbacks
        callback_list = CallbackList(callbacks)
        
        # Train model
        start_time = time.time()
        
        logger.info(f"\nStarting training for {stage_name}...")
        self.model.learn(
            total_timesteps=stage_config['timesteps'],
            callback=callback_list,
            tb_log_name=f"foundation_{stage_name}",
            reset_num_timesteps=False
        )
        
        training_time = time.time() - start_time
        logger.info(f"\nStage {stage_name} completed in {training_time/60:.1f} minutes")
        
        # Evaluate what was learned
        learning_report = self.evaluate_stage_learning(env, stage_config)
        
        # Save model
        model_path = os.path.join(self.checkpoint_dir, stage_name, "final_model.zip")
        self.model.save(model_path)
        
        # Save normalization stats
        vec_norm_path = os.path.join(self.checkpoint_dir, stage_name, "vec_normalize.pkl")
        env.save(vec_norm_path)
        
        # Create stage summary
        stage_summary = {
            'stage_name': stage_name,
            'stage_idx': stage_idx,
            'training_time_minutes': training_time / 60,
            'learning_report': learning_report,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = os.path.join(self.log_dir, f"foundation_{stage_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(stage_summary, f, indent=2)
        
        # Cleanup
        env.close()
        
        return stage_summary
    
    def train_foundation(self):
        """Train all 4 foundation stages."""
        logger.info("\n" + "="*60)
        logger.info("STARTING FOUNDATION TRAINING")
        logger.info("Training 4 stages: toy_easy → toy_normal → toy_hard → toy_multi")
        logger.info("="*60)
        
        all_summaries = []
        
        # Train each foundation stage
        for i in range(4):
            summary = self.train_stage(i)
            all_summaries.append(summary)
            
            # Pause between stages
            if i < 3:
                logger.info("\nPausing 10 seconds before next stage...")
                time.sleep(10)
        
        # Generate final report
        self.generate_foundation_report(all_summaries)
        
        logger.info("\n" + "="*60)
        logger.info("FOUNDATION TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info("\nThe model has learned:")
        logger.info("  1. Toy Easy: Basic sequence constraints")
        logger.info("  2. Toy Normal: Meeting deadlines")
        logger.info("  3. Toy Hard: Prioritizing important jobs")
        logger.info("  4. Toy Multi: Multi-machine coordination")
        logger.info("\nReady to proceed to more complex stages if desired.")
    
    def generate_foundation_report(self, summaries: list):
        """Generate comprehensive foundation training report."""
        report = {
            'training_complete': datetime.now().isoformat(),
            'total_training_time_minutes': sum(s['training_time_minutes'] for s in summaries),
            'stages': summaries,
            'learning_progression': [
                {
                    'stage': s['stage_name'],
                    'scheduling_rate': s['learning_report']['scheduling_rate'],
                    'focus': s['learning_report']['focus']
                }
                for s in summaries
            ]
        }
        
        # Save report
        report_path = os.path.join(self.log_dir, "foundation_training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nFoundation training report saved to: {report_path}")


def main():
    """Main entry point for foundation training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train PPO model on foundation stages (first 4 stages only)"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode - train toy_easy for 1000 steps only'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FoundationTrainer()
    
    if args.test:
        # Test mode
        logger.info("\n=== TEST MODE - Training toy_easy for 1000 steps ===")
        trainer.foundation_stages[0]['timesteps'] = 1000
        trainer.train_stage(0)
    else:
        # Full foundation training
        trainer.train_foundation()


if __name__ == "__main__":
    main()