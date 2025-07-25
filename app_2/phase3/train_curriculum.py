"""
Phase 3: Curriculum Learning Training Script

Progressive training from toy to production scale using PPO.
"""

import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.curriculum_env import CurriculumSchedulingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurriculumTrainingCallback(BaseCallback):
    """Custom callback for curriculum training progress."""
    
    def __init__(self, stage_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            
            # Record episode metrics
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
            if 'episode_metrics' in info:
                self.episode_metrics.append(info['episode_metrics'])
                
                # Log metrics
                metrics = info['episode_metrics']
                logger.info(
                    f"[{self.stage_name}] Episode {len(self.episode_rewards)} - "
                    f"Reward: {self.episode_rewards[-1]:.2f}, "
                    f"Completed: {metrics['jobs_completed']}, "
                    f"Late: {metrics['jobs_late']}, "
                    f"Utilization: {metrics['machine_utilization']:.1%}"
                )
                
        return True
        
    def _on_training_end(self) -> None:
        # Summary statistics
        if self.episode_rewards:
            logger.info(
                f"[{self.stage_name}] Training Summary - "
                f"Episodes: {len(self.episode_rewards)}, "
                f"Avg Reward: {np.mean(self.episode_rewards):.2f}, "
                f"Max Reward: {np.max(self.episode_rewards):.2f}"
            )


class CurriculumTrainer:
    """Manages curriculum learning training process."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup directories
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        self.tensorboard_dir = self.config['training']['tensorboard_log']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Device configuration
        if self.config['training']['device'] == 'mps' and torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) for training")
            self.device = 'mps'
        else:
            self.device = 'cpu'
            logger.warning("MPS not available, using CPU")
            
        # Snapshot paths
        self.snapshot_dir = 'phase3/snapshots'
        self.snapshot_paths = {
            'snapshot_normal': os.path.join(self.snapshot_dir, 'snapshot_normal.json'),
            'snapshot_rush': os.path.join(self.snapshot_dir, 'snapshot_rush.json'),
            'snapshot_heavy': os.path.join(self.snapshot_dir, 'snapshot_heavy.json'),
            'snapshot_bottleneck': os.path.join(self.snapshot_dir, 'snapshot_bottleneck.json'),
            'snapshot_multi_heavy': os.path.join(self.snapshot_dir, 'snapshot_multi_heavy.json'),
            'edge_same_machine': os.path.join(self.snapshot_dir, 'edge_case_same_machine.json'),
            'edge_cascading': os.path.join(self.snapshot_dir, 'edge_case_cascading.json'),
            'edge_conflicts': os.path.join(self.snapshot_dir, 'edge_case_conflicts.json'),
            'edge_multi_complex': os.path.join(self.snapshot_dir, 'edge_case_multi_complex.json')
        }
        
        # Training stages in order
        self.stages = [
            # Foundation
            'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
            # Strategy
            'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
            # Scale
            'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
            # Production
            'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
        ]
        
        # Current model (persists across stages)
        self.model = None
        self.current_stage_idx = 0
        
    def create_env(self, stage_name: str, eval_mode: bool = False) -> gym.Env:
        """Create environment for given stage."""
        stage_config = self.config[stage_name]
        
        # Determine snapshot path
        snapshot_path = None
        data_source = stage_config['data_source']
        
        if data_source != 'synthetic':
            # Map data source to snapshot
            if 'rush' in data_source:
                snapshot_path = self.snapshot_paths['snapshot_rush']
            elif 'heavy' in data_source:
                snapshot_path = self.snapshot_paths['snapshot_heavy']
            elif 'bottleneck' in data_source:
                snapshot_path = self.snapshot_paths['snapshot_bottleneck']
            elif 'edge_same_machine' in data_source:
                snapshot_path = self.snapshot_paths['edge_same_machine']
            elif 'edge_cascading' in data_source:
                snapshot_path = self.snapshot_paths['edge_cascading']
            elif 'mixed' in data_source:
                # For expert stage, randomly select snapshots
                snapshot_path = np.random.choice(list(self.snapshot_paths.values()))
            else:
                snapshot_path = self.snapshot_paths['snapshot_normal']
                
        # Create environment
        env = CurriculumSchedulingEnv(
            stage_config=stage_config,
            snapshot_path=snapshot_path,
            reward_profile='balanced',  # Can vary this too
            seed=42 if eval_mode else None
        )
        
        # Wrap with Monitor
        env = Monitor(env)
        
        return env
        
    def train_stage(self, stage_name: str):
        """Train a single curriculum stage."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage: {stage_name}")
        logger.info(f"{'='*60}")
        
        stage_config = self.config[stage_name]
        
        # Create training environments
        n_envs = self.config['training']['n_envs']
        
        def make_env():
            return self.create_env(stage_name)
            
        # Vectorized environments
        train_env = make_vec_env(
            make_env,
            n_envs=n_envs,
            seed=42
        )
        
        # Normalize observations and rewards
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0
        )
        
        # Always create new model for each stage (observation space changes)
        logger.info("Creating new PPO model for stage")
        
        self.model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=float(stage_config['learning_rate']),
            n_steps=int(stage_config['n_steps']),
            batch_size=int(stage_config['batch_size']),
            n_epochs=10,
            gamma=self.config['training']['gamma'],
            gae_lambda=self.config['training']['gae_lambda'],
            clip_range=self.config['training']['clip_range'],
            vf_coef=self.config['training']['vf_coef'],
            ent_coef=self.config['training']['ent_coef'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            tensorboard_log=os.path.join(self.tensorboard_dir, stage_name),
            device=self.device,
            verbose=self.config['training']['verbose']
        )
            
        # Callbacks
        callbacks = []
        
        # Custom progress callback
        progress_callback = CurriculumTrainingCallback(stage_name)
        callbacks.append(progress_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training']['checkpoint_freq'],
            save_path=os.path.join(self.checkpoint_dir, stage_name),
            name_prefix=f'ppo_{stage_name}'
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_env(stage_name, eval_mode=True)
        # Wrap eval env in VecNormalize to match training env
        from stable_baselines3.common.vec_env import DummyVecEnv
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=self.config['training']['n_eval_episodes'],
            eval_freq=self.config['training']['eval_freq'],
            best_model_save_path=os.path.join(self.checkpoint_dir, stage_name),
            deterministic=True
        )
        callbacks.append(eval_callback)
        
        # Combine callbacks
        callback_list = CallbackList(callbacks)
        
        # Train
        logger.info(f"Training for {stage_config['timesteps']} timesteps")
        
        self.model.learn(
            total_timesteps=stage_config['timesteps'],
            callback=callback_list,
            reset_num_timesteps=False,  # Continue from previous stage
            progress_bar=True
        )
        
        # Save final model for stage
        model_path = os.path.join(self.checkpoint_dir, f'{stage_name}_final.zip')
        self.model.save(model_path)
        
        # Save normalization statistics
        vec_norm_path = os.path.join(self.checkpoint_dir, f'{stage_name}_vec_normalize.pkl')
        train_env.save(vec_norm_path)
        
        logger.info(f"Stage {stage_name} completed!")
        
        # Cleanup
        train_env.close()
        eval_env.close()
        
    def train_curriculum(self, start_stage: Optional[str] = None):
        """Train through entire curriculum."""
        # Find starting point
        if start_stage:
            try:
                self.current_stage_idx = self.stages.index(start_stage)
                logger.info(f"Starting from stage: {start_stage}")
                
                # Load previous model if not first stage
                if self.current_stage_idx > 0:
                    prev_stage = self.stages[self.current_stage_idx - 1]
                    model_path = os.path.join(self.checkpoint_dir, f'{prev_stage}_final.zip')
                    
                    if os.path.exists(model_path):
                        logger.info(f"Loading model from previous stage: {prev_stage}")
                        self.model = PPO.load(model_path, device=self.device)
                        
            except ValueError:
                logger.error(f"Unknown stage: {start_stage}")
                return
                
        # Train through stages
        for stage_idx in range(self.current_stage_idx, len(self.stages)):
            stage_name = self.stages[stage_idx]
            
            try:
                self.train_stage(stage_name)
                self.current_stage_idx = stage_idx + 1
                
                # Save progress
                self.save_training_state()
                
            except Exception as e:
                logger.error(f"Error in stage {stage_name}: {e}")
                raise
                
        logger.info("\n" + "="*60)
        logger.info("CURRICULUM TRAINING COMPLETED!")
        logger.info("="*60)
        
    def save_training_state(self):
        """Save current training state."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'completed_stages': self.stages[:self.current_stage_idx],
            'timestamp': datetime.now().isoformat()
        }
        
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_training_state(self) -> Optional[Dict[str, Any]]:
        """Load previous training state."""
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                return json.load(f)
                
        return None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Phase 3 Curriculum Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase3_curriculum_config.yaml',
        help='Path to curriculum configuration'
    )
    parser.add_argument(
        '--start-stage',
        type=str,
        default=None,
        help='Stage to start/resume from'
    )
    parser.add_argument(
        '--single-stage',
        type=str,
        default=None,
        help='Train only a single stage'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CurriculumTrainer(args.config)
    
    # Check for previous state
    if not args.start_stage:
        state = trainer.load_training_state()
        if state:
            logger.info(f"Found previous training state: {state['completed_stages']}")
            # Resume from next stage
            if state['current_stage_idx'] < len(trainer.stages):
                args.start_stage = trainer.stages[state['current_stage_idx']]
                
    # Train
    if args.single_stage:
        # Train single stage only
        trainer.train_stage(args.single_stage)
    else:
        # Full curriculum
        trainer.train_curriculum(start_stage=args.start_stage)


if __name__ == '__main__':
    main()