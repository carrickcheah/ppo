"""
Phase 3 Curriculum Training Script
Implements 16-stage progressive training for PPO scheduler
"""

import os
import sys
import yaml
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from environments.curriculum_env import CurriculumSchedulingEnv
# For now, we'll use MlpPolicy until we implement the transformer policy
# from src.model.ppo_scheduler import PPOScheduler
# from src.model.transformer_policy import TransformerSchedulingPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurriculumTrainingCallback(BaseCallback):
    """Custom callback for curriculum training monitoring."""
    
    def __init__(self, stage_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.episode_rewards = []
        self.episode_metrics = []
        self.best_reward = -float('inf')
        self.best_utilization = 0.0
        
    def _on_step(self) -> bool:
        # Track episode completions
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            if 'final_metrics' in info:
                metrics = info['final_metrics']
                reward = self.locals.get('rewards')[0]
                
                self.episode_rewards.append(reward)
                self.episode_metrics.append(metrics)
                
                # Update best metrics
                utilization = metrics.get('machine_utilization', 0)
                if reward > self.best_reward:
                    self.best_reward = reward
                if utilization > self.best_utilization:
                    self.best_utilization = utilization
                
                # Log progress
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_utilization = np.mean([m['machine_utilization'] for m in self.episode_metrics[-10:]])
                    avg_jobs_completed = np.mean([m['jobs_completed'] for m in self.episode_metrics[-10:]])
                    avg_late = np.mean([m['jobs_late'] for m in self.episode_metrics[-10:]])
                    
                    logger.info(
                        f"[{self.stage_name}] Episodes: {len(self.episode_rewards)}, "
                        f"Avg Reward: {avg_reward:.2f}, Best: {self.best_reward:.2f}, "
                        f"Util: {avg_utilization:.1%}, Jobs: {avg_jobs_completed:.1f}, "
                        f"Late: {avg_late:.1f}"
                    )
        
        return True


class CurriculumTrainer:
    """Manages the 16-stage curriculum training process."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        self.log_dir = self.config['training']['log_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize stage tracking
        self.current_stage_idx = 0
        self.stages = self.config['curriculum']['stages']
        self.stage_history = []
        
        # Model and environment
        self.model = None
        self.env = None
        
    def create_env(self, stage_config: Dict[str, Any], seed: int = 42) -> DummyVecEnv:
        """Create environment for a specific stage."""
        def make_env():
            env = CurriculumSchedulingEnv(
                stage_config=stage_config,
                data_source="synthetic",
                reward_profile=stage_config.get('reward_profile', 'learning'),
                seed=seed
            )
            env = Monitor(env)
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env])
        
        # Add normalization
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        return env
    
    def create_model(self, env: VecNormalize, stage_config: Dict[str, Any]) -> PPO:
        """Create or update PPO model."""
        # Get hyperparameters
        hyperparams = self.config['hyperparameters'].copy()
        
        # Stage-specific adjustments
        if 'hyperparameter_overrides' in stage_config:
            hyperparams.update(stage_config['hyperparameter_overrides'])
        
        # Adjust entropy coefficient for exploration
        if stage_config.get('name', '').startswith('toy'):
            hyperparams['ent_coef'] = 0.05  # More exploration for toy stages
        
        # For curriculum learning, we need to create a new model for each stage
        # due to changing observation/action spaces
        logger.info(f"Creating new PPO model with hyperparameters: {hyperparams}")
        
        model = PPO(
            policy="MlpPolicy",  # Using MLP for now
            env=env,
            learning_rate=float(hyperparams['learning_rate']),
            n_steps=hyperparams['n_steps'],
            batch_size=hyperparams['batch_size'],
            n_epochs=hyperparams['n_epochs'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            clip_range=hyperparams['clip_range'],
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            tensorboard_log=os.path.join(self.log_dir, 'tensorboard'),
            # policy_kwargs={
            #     'transformer_config': self.config['model']['transformer_config']
            # },
            verbose=1
        )
        
        return model
    
    def train_stage(self, stage_idx: int) -> Dict[str, Any]:
        """Train on a single curriculum stage."""
        stage_config = self.stages[stage_idx]
        stage_name = stage_config['name']
        timesteps = stage_config.get('timesteps', 100000)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage {stage_idx + 1}/{len(self.stages)}: {stage_name}")
        logger.info(f"Jobs: {stage_config['jobs']}, Machines: {stage_config['machines']}")
        logger.info(f"Description: {stage_config['description']}")
        logger.info(f"Training for {timesteps:,} timesteps")
        logger.info(f"{'='*60}\n")
        
        # Create environment
        self.env = self.create_env(stage_config)
        
        # Create or update model
        self.model = self.create_model(self.env, stage_config)
        
        # Setup callbacks
        callbacks = []
        
        # Custom monitoring callback
        monitor_callback = CurriculumTrainingCallback(stage_name)
        callbacks.append(monitor_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.checkpoint_dir, stage_name),
            name_prefix=f"ppo_{stage_name}"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_env(stage_config, seed=123)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.checkpoint_dir, stage_name, 'best'),
            log_path=os.path.join(self.log_dir, stage_name),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Train
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=stage_name
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save final model
        model_path = os.path.join(self.checkpoint_dir, stage_name, f"{stage_name}_final.zip")
        self.model.save(model_path)
        
        # Save normalization stats
        norm_path = os.path.join(self.checkpoint_dir, stage_name, f"{stage_name}_vecnorm.pkl")
        self.env.save(norm_path)
        
        # Compile stage results
        results = {
            'stage_name': stage_name,
            'stage_idx': stage_idx,
            'timesteps': timesteps,
            'training_time': training_time,
            'best_reward': monitor_callback.best_reward,
            'best_utilization': monitor_callback.best_utilization,
            'final_metrics': monitor_callback.episode_metrics[-1] if monitor_callback.episode_metrics else {},
            'model_path': model_path,
            'norm_path': norm_path
        }
        
        # Check performance targets
        if 'performance_targets' in stage_config:
            targets = stage_config['performance_targets']
            passed = True
            
            if 'min_utilization' in targets:
                if results['best_utilization'] < targets['min_utilization']:
                    logger.warning(
                        f"Failed utilization target: {results['best_utilization']:.1%} < "
                        f"{targets['min_utilization']:.1%}"
                    )
                    passed = False
            
            if 'max_late_ratio' in targets and results['final_metrics']:
                jobs_completed = results['final_metrics'].get('jobs_completed', 1)
                jobs_late = results['final_metrics'].get('jobs_late', 0)
                late_ratio = jobs_late / max(1, jobs_completed)
                
                if late_ratio > targets['max_late_ratio']:
                    logger.warning(
                        f"Failed late ratio target: {late_ratio:.1%} > "
                        f"{targets['max_late_ratio']:.1%}"
                    )
                    passed = False
            
            results['passed_targets'] = passed
        else:
            results['passed_targets'] = True
        
        logger.info(f"\nStage {stage_name} completed!")
        logger.info(f"Best reward: {results['best_reward']:.2f}")
        logger.info(f"Best utilization: {results['best_utilization']:.1%}")
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        
        return results
    
    def run_curriculum(self, start_stage: int = 0):
        """Run the full curriculum training."""
        logger.info("\n" + "="*80)
        logger.info("STARTING 16-STAGE CURRICULUM TRAINING")
        logger.info("="*80 + "\n")
        
        # Training loop
        for stage_idx in range(start_stage, len(self.stages)):
            self.current_stage_idx = stage_idx
            
            # Train stage
            results = self.train_stage(stage_idx)
            self.stage_history.append(results)
            
            # Save progress
            progress_path = os.path.join(self.checkpoint_dir, 'curriculum_progress.json')
            # Convert numpy types to Python types for JSON serialization
            serializable_history = []
            for result in self.stage_history:
                serializable_result = {}
                for k, v in result.items():
                    if isinstance(v, (np.floating, np.integer)):
                        serializable_result[k] = float(v)
                    elif isinstance(v, dict):
                        serializable_result[k] = {
                            k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
                            for k2, v2 in v.items()
                        }
                    else:
                        serializable_result[k] = v
                serializable_history.append(serializable_result)
            
            with open(progress_path, 'w') as f:
                json.dump({
                    'current_stage': stage_idx + 1,
                    'total_stages': len(self.stages),
                    'stage_history': serializable_history
                }, f, indent=2)
            
            # Check if we should continue
            if not results['passed_targets']:
                logger.warning(f"\nStage {results['stage_name']} did not meet performance targets.")
                logger.warning("Consider adjusting hyperparameters or training longer.")
                
                # Optional: retry logic here
                retry = input("Retry this stage? (y/n): ").lower() == 'y'
                if retry:
                    stage_idx -= 1  # Retry same stage
                    continue
        
        logger.info("\n" + "="*80)
        logger.info("CURRICULUM TRAINING COMPLETE!")
        logger.info("="*80 + "\n")
        
        # Summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of the training."""
        report_path = os.path.join(self.checkpoint_dir, 'training_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("CURRICULUM TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            total_time = sum(r['training_time'] for r in self.stage_history)
            f.write(f"Total training time: {total_time/3600:.1f} hours\n")
            f.write(f"Stages completed: {len(self.stage_history)}/{len(self.stages)}\n\n")
            
            f.write("Stage Results:\n")
            f.write("-" * 60 + "\n")
            
            for result in self.stage_history:
                f.write(f"\n{result['stage_name']}:\n")
                f.write(f"  Best reward: {result['best_reward']:.2f}\n")
                f.write(f"  Best utilization: {result['best_utilization']:.1%}\n")
                f.write(f"  Training time: {result['training_time']/60:.1f} minutes\n")
                f.write(f"  Passed targets: {'Yes' if result['passed_targets'] else 'No'}\n")
                
                if result['final_metrics']:
                    metrics = result['final_metrics']
                    f.write(f"  Jobs completed: {metrics.get('jobs_completed', 0)}/{result['stage_name'].split()[1]}\n")
                    f.write(f"  Jobs late: {metrics.get('jobs_late', 0)}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Training complete! Model ready for deployment.\n")
        
        logger.info(f"Summary report saved to: {report_path}")
        
        # Also save as JSON for easier parsing
        json_report_path = os.path.join(self.checkpoint_dir, 'training_summary.json')
        with open(json_report_path, 'w') as f:
            json.dump({
                'total_training_time_hours': total_time / 3600,
                'stages_completed': len(self.stage_history),
                'total_stages': len(self.stages),
                'stage_results': self.stage_history,
                'final_model_path': self.stage_history[-1]['model_path'] if self.stage_history else None
            }, f, indent=2)


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 3 Curriculum Training")
    parser.add_argument(
        '--config',
        type=str,
        default='/Users/carrickcheah/Project/ppo/app_2/configs/phase3_curriculum_config.yaml',
        help='Path to curriculum config file'
    )
    parser.add_argument(
        '--start-stage',
        type=int,
        default=0,
        help='Stage to start from (0-based index)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    # Create trainer
    trainer = CurriculumTrainer(args.config)
    
    # Handle resume
    start_stage = args.start_stage
    if args.resume:
        progress_path = os.path.join(trainer.checkpoint_dir, 'curriculum_progress.json')
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                start_stage = progress['current_stage']
                trainer.stage_history = progress.get('stage_history', [])
                logger.info(f"Resuming from stage {start_stage + 1}")
    
    # Run training
    try:
        trainer.run_curriculum(start_stage=start_stage)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        logger.info(f"Progress saved. Resume with --resume --start-stage {trainer.current_stage_idx}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()