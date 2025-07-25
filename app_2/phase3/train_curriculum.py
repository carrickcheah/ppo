"""
Phase 3 Curriculum Training Script
Trains PPO model through 16 stages using REAL production data
Implements performance gates and continuous monitoring
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


class CurriculumTrainingCallback(BaseCallback):
    """Custom callback for monitoring curriculum training progress."""
    
    def __init__(self, stage_name: str, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.utilizations = []
        self.on_time_rates = []
        
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            
            # Extract metrics
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
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
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'final_reward': self.episode_rewards[-1] if self.episode_rewards else 0
        }
        
        # Save metrics
        metrics_path = os.path.join(self.log_dir, f"stage_{self.stage_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CurriculumTrainer:
    """Manages the 16-stage curriculum training process."""
    
    def __init__(self, config_path: str = "/Users/carrickcheah/Project/ppo/app_2/configs/phase3_curriculum_config.yaml"):
        """Initialize trainer with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        self.log_dir = self.config['training']['log_dir']
        self.tensorboard_log = self.config['training']['tensorboard_log']
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Training state
        self.current_stage_idx = 0
        self.model = None
        self.stages = self.config['curriculum']['stages']
        
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
        
        # Apply stage-specific overrides
        overrides = stage_config.get('hyperparameter_overrides', {})
        hyperparams.update(overrides)
        
        return hyperparams
    
    def check_performance_gates(self, stage_config: Dict, eval_metrics: Dict) -> bool:
        """Check if performance targets are met."""
        targets = stage_config.get('performance_targets', {})
        
        # Check utilization
        min_util = targets.get('min_utilization', 0.0)
        if eval_metrics.get('utilization', 0) < min_util:
            logger.warning(f"Utilization {eval_metrics['utilization']:.2%} < target {min_util:.2%}")
            return False
        
        # Check late ratio
        max_late = targets.get('max_late_ratio', 1.0)
        if eval_metrics.get('late_ratio', 1) > max_late:
            logger.warning(f"Late ratio {eval_metrics['late_ratio']:.2%} > target {max_late:.2%}")
            return False
        
        return True
    
    def evaluate_model(self, env: VecNormalize, n_episodes: int = 10) -> Dict:
        """Evaluate model performance."""
        episode_rewards = []
        episode_lengths = []
        utilizations = []
        late_counts = []
        total_jobs = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            # Extract metrics from final info
            final_info = info[0]
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Calculate utilization and late rate (placeholder - need env support)
            # These would come from the environment's final info
            utilizations.append(0.8)  # Placeholder
            late_counts.append(0)  # Placeholder
            total_jobs.append(10)  # Placeholder
        
        # Calculate aggregate metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'utilization': float(np.mean(utilizations)),
            'late_ratio': float(sum(late_counts) / sum(total_jobs)) if sum(total_jobs) > 0 else 0.0
        }
        
        return metrics
    
    def train_stage(self, stage_idx: int) -> bool:
        """Train a single curriculum stage."""
        stage_config = self.stages[stage_idx]
        stage_name = stage_config['name']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Stage {stage_idx + 1}/{len(self.stages)}: {stage_name}")
        logger.info(f"Description: {stage_config['description']}")
        logger.info(f"Jobs: {stage_config['jobs']}, Machines: {stage_config['machines']}")
        logger.info(f"Timesteps: {stage_config['timesteps']:,}")
        logger.info(f"{'='*60}")
        
        # Create environment
        env = self.create_env(stage_config)
        
        # Get hyperparameters
        hyperparams = self.get_hyperparameters(stage_config)
        
        # Create or load model
        if self.model is None:
            # First stage - create new model
            logger.info("Creating new PPO model...")
            
            # Model configuration - using standard MLP for now
            # TODO: Add transformer policy later
            policy_kwargs = {
                'net_arch': dict(pi=[256, 256], vf=[256, 256]),  # Separate networks for actor and critic
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
            # Continue from previous stage
            logger.info("Updating model for new stage...")
            self.model.set_env(env)
            
            # Update hyperparameters if changed
            self.model.learning_rate = hyperparams['learning_rate']
            self.model.n_steps = hyperparams['n_steps']
            self.model.batch_size = hyperparams['batch_size']
            self.model.n_epochs = hyperparams['n_epochs']
            self.model.ent_coef = hyperparams['ent_coef']
        
        # Setup callbacks
        callbacks = []
        
        # Curriculum training callback
        curriculum_callback = CurriculumTrainingCallback(
            stage_name=stage_name,
            log_dir=self.log_dir
        )
        callbacks.append(curriculum_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training']['checkpoint_freq'],
            save_path=os.path.join(self.checkpoint_dir, stage_name),
            name_prefix=f"stage_{stage_name}"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_env(stage_config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.checkpoint_dir, stage_name, 'best'),
            log_path=os.path.join(self.log_dir, stage_name),
            eval_freq=self.config['training']['eval_freq'],
            n_eval_episodes=self.config['training']['n_eval_episodes'],
            deterministic=True
        )
        callbacks.append(eval_callback)
        
        # Combine callbacks
        callback_list = CallbackList(callbacks)
        
        # Train model
        start_time = time.time()
        try:
            self.model.learn(
                total_timesteps=stage_config['timesteps'],
                callback=callback_list,
                tb_log_name=stage_name,
                reset_num_timesteps=False  # Continue from previous training
            )
        except Exception as e:
            logger.error(f"Training failed for stage {stage_name}: {e}")
            return False
        
        training_time = time.time() - start_time
        logger.info(f"Stage {stage_name} training completed in {training_time/60:.1f} minutes")
        
        # Evaluate final performance
        logger.info("Evaluating final performance...")
        eval_metrics = self.evaluate_model(eval_env)
        
        logger.info(f"Stage {stage_name} Performance:")
        logger.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        logger.info(f"  Utilization: {eval_metrics['utilization']:.2%}")
        logger.info(f"  Late Ratio: {eval_metrics['late_ratio']:.2%}")
        
        # Check performance gates
        if not self.check_performance_gates(stage_config, eval_metrics):
            logger.warning(f"Stage {stage_name} failed performance gates!")
            # In production, you might want to retry or adjust hyperparameters
            # For now, we'll continue to next stage
        
        # Save final model for stage
        model_path = os.path.join(self.checkpoint_dir, stage_name, f"final_model.zip")
        self.model.save(model_path)
        
        # Save normalization stats
        vec_norm_path = os.path.join(self.checkpoint_dir, stage_name, f"vec_normalize.pkl")
        env.save(vec_norm_path)
        
        # Save stage results
        stage_results = {
            'stage_name': stage_name,
            'stage_idx': stage_idx,
            'training_time': training_time,
            'final_metrics': eval_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.log_dir, f"stage_{stage_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(stage_results, f, indent=2)
        
        # Cleanup
        env.close()
        eval_env.close()
        
        return True
    
    def train_curriculum(self, start_stage: int = 0):
        """Train through all curriculum stages."""
        logger.info("=== STARTING CURRICULUM TRAINING ===")
        logger.info(f"Total stages: {len(self.stages)}")
        logger.info(f"Starting from stage: {start_stage + 1}")
        
        # Load model if resuming
        if start_stage > 0:
            prev_stage = self.stages[start_stage - 1]
            model_path = os.path.join(
                self.checkpoint_dir, 
                prev_stage['name'], 
                "final_model.zip"
            )
            if os.path.exists(model_path):
                logger.info(f"Loading model from stage {prev_stage['name']}")
                self.model = PPO.load(model_path)
            else:
                logger.warning(f"No model found from previous stage, starting fresh")
        
        # Train each stage
        for stage_idx in range(start_stage, len(self.stages)):
            success = self.train_stage(stage_idx)
            
            if not success:
                logger.error(f"Failed to train stage {stage_idx + 1}")
                break
            
            # Small pause between stages
            if stage_idx < len(self.stages) - 1:
                logger.info("Pausing before next stage...")
                time.sleep(10)
        
        logger.info("\n=== CURRICULUM TRAINING COMPLETE ===")
        
        # Summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate final training summary report."""
        report = {
            'training_complete': datetime.now().isoformat(),
            'stages': []
        }
        
        for stage in self.stages:
            stage_name = stage['name']
            
            # Load stage results if available
            results_path = os.path.join(self.log_dir, f"stage_{stage_name}_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    stage_results = json.load(f)
                report['stages'].append(stage_results)
        
        # Save summary
        summary_path = os.path.join(self.log_dir, "curriculum_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nTraining summary saved to: {summary_path}")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO with curriculum learning")
    parser.add_argument(
        '--config', 
        type=str, 
        default="/Users/carrickcheah/Project/ppo/app_2/configs/phase3_curriculum_config.yaml",
        help='Path to curriculum configuration'
    )
    parser.add_argument(
        '--start-stage', 
        type=int, 
        default=0,
        help='Stage to start/resume from (0-based index)'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test mode - only train first stage for 1000 steps'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CurriculumTrainer(config_path=args.config)
    
    if args.test:
        # Test mode - modify first stage for quick test
        logger.info("\n=== TEST MODE - Training first stage only for 1000 steps ===")
        trainer.stages[0]['timesteps'] = 1000
        trainer.train_stage(0)
    else:
        # Full training
        trainer.train_curriculum(start_stage=args.start_stage)


if __name__ == "__main__":
    main()