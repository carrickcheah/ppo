"""
Curriculum learning trainer for progressive difficulty stages.
Trains PPO model from simple to complex scheduling problems.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.scheduling_env import SchedulingEnv
from models.ppo_scheduler import PPOScheduler, PPOConfig
from models.rollout_buffer import RolloutBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    name: str
    data_path: str
    n_timesteps: int
    success_threshold: float = 0.8
    min_reward: float = 0.0
    description: str = ""


class CurriculumTrainer:
    """Manages curriculum learning across multiple stages."""
    
    def __init__(
        self,
        stages: List[StageConfig],
        checkpoint_dir: str = "checkpoints/curriculum",
        tensorboard_dir: str = "tensorboard/curriculum",
        config: Optional[PPOConfig] = None
    ):
        """
        Initialize curriculum trainer.
        
        Args:
            stages: List of stage configurations
            checkpoint_dir: Directory for checkpoints
            tensorboard_dir: Directory for tensorboard logs
            config: PPO configuration
        """
        self.stages = stages
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.config = config or PPOConfig()
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Initialize tracking
        self.current_stage = 0
        self.stage_results = []
        self.global_timesteps = 0
        
        # Create tensorboard writer
        run_name = f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(os.path.join(tensorboard_dir, run_name))
        
        logger.info(f"Initialized curriculum trainer with {len(stages)} stages")
        
    def train_stage(
        self,
        stage: StageConfig,
        ppo: Optional[PPOScheduler] = None,
        stage_idx: int = 0
    ) -> Tuple[PPOScheduler, Dict]:
        """
        Train a single curriculum stage.
        
        Args:
            stage: Stage configuration
            ppo: Optional PPO model to continue from
            stage_idx: Stage index
            
        Returns:
            (trained_ppo, metrics)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Stage {stage_idx + 1}: {stage.name}")
        logger.info(f"Data: {stage.data_path}")
        logger.info(f"Target timesteps: {stage.n_timesteps}")
        logger.info(f"Success threshold: {stage.success_threshold:.1%}")
        logger.info(f"{'='*50}\n")
        
        # Create environment with longer episodes for complex stages
        max_steps = 1500 if stage_idx <= 2 else 2500  # More steps for larger problems
        env = SchedulingEnv(stage.data_path, max_steps=max_steps)
        logger.info(f"Environment: {env.n_tasks} tasks, {env.n_machines} machines")
        
        # Create new PPO model for each stage (observation dimensions change)
        # Note: In a production setting, we might want to transfer weights for shared layers
        # but for now we'll create fresh models for each stage
        ppo = PPOScheduler(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=self.config.learning_rate * (0.9 ** stage_idx),  # Decay LR for later stages
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            clip_range=self.config.clip_range,
            value_loss_coef=self.config.value_loss_coef,
            entropy_coef=self.config.entropy_coef,
            max_grad_norm=self.config.max_grad_norm,
            gae_lambda=self.config.gae_lambda,
            gamma=self.config.gamma,
            device=self.config.device
        )
        
        if stage_idx == 0:
            logger.info("Created new PPO model for first stage")
        else:
            logger.info(f"Created new PPO model for stage {stage_idx + 1} (obs_dim={env.observation_space.shape[0]}, LR={ppo.learning_rate:.2e})")
            
        # Create rollout buffer
        buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        
        # Training metrics
        best_reward = -float('inf')
        best_success_rate = 0.0
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        
        # Training loop
        timesteps = 0
        updates = 0
        pbar = tqdm(total=stage.n_timesteps, desc=f"Stage {stage_idx + 1}")
        
        while timesteps < stage.n_timesteps:
            # Collect rollouts
            rollout_stats = self._collect_rollouts(
                env, ppo, buffer, self.config.n_steps
            )
            
            # Update model
            train_stats = ppo.train_on_buffer(buffer, self.writer)
            
            # Update counters
            timesteps += self.config.n_steps
            self.global_timesteps += self.config.n_steps
            updates += 1
            
            # Track metrics
            episode_rewards.extend(rollout_stats['episode_rewards'])
            episode_lengths.extend(rollout_stats['episode_lengths'])
            episode_success.extend(rollout_stats['episode_success'])
            
            # Calculate current performance
            if len(episode_rewards) > 0:
                recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                mean_reward = np.mean(recent_rewards)
                recent_success = episode_success[-10:] if len(episode_success) >= 10 else episode_success
                success_rate = np.mean(recent_success)
                
                # Log to tensorboard
                self.writer.add_scalar(f'stage_{stage_idx}/mean_reward', mean_reward, self.global_timesteps)
                self.writer.add_scalar(f'stage_{stage_idx}/success_rate', success_rate, self.global_timesteps)
                self.writer.add_scalar(f'stage_{stage_idx}/mean_length', rollout_stats['mean_length'], self.global_timesteps)
                
                # Update progress bar
                pbar.update(self.config.n_steps)
                pbar.set_postfix({
                    'reward': f"{mean_reward:.1f}",
                    'success': f"{success_rate:.1%}",
                    'episodes': len(episode_rewards)
                })
                
                # Check for best model
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_success_rate = success_rate
                    
                    # Save best model for stage
                    best_path = os.path.join(
                        self.checkpoint_dir,
                        f"stage_{stage_idx}_{stage.name}_best.pth"
                    )
                    ppo.save(best_path)
                    
                # Early stopping if success threshold met
                if success_rate >= stage.success_threshold and mean_reward >= stage.min_reward:
                    logger.info(f"Stage {stage_idx + 1} success threshold reached!")
                    logger.info(f"Success rate: {success_rate:.1%}, Mean reward: {mean_reward:.1f}")
                    break
                    
            # Clear buffer
            buffer.reset()
            
        pbar.close()
        
        # Save final model for stage
        final_path = os.path.join(
            self.checkpoint_dir,
            f"stage_{stage_idx}_{stage.name}_final.pth"
        )
        ppo.save(final_path)
        
        # Compile stage metrics
        metrics = {
            'stage_name': stage.name,
            'timesteps': timesteps,
            'episodes': len(episode_rewards),
            'best_reward': best_reward,
            'best_success_rate': best_success_rate,
            'final_reward': np.mean(episode_rewards[-10:]) if episode_rewards else 0,
            'final_success_rate': np.mean(episode_success[-10:]) if episode_success else 0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0
        }
        
        logger.info(f"\nStage {stage_idx + 1} Complete:")
        logger.info(f"  Best reward: {best_reward:.1f}")
        logger.info(f"  Best success rate: {best_success_rate:.1%}")
        logger.info(f"  Final success rate: {metrics['final_success_rate']:.1%}")
        
        return ppo, metrics
        
    def _collect_rollouts(
        self,
        env: SchedulingEnv,
        ppo: PPOScheduler,
        buffer: RolloutBuffer,
        n_steps: int
    ) -> Dict:
        """Collect rollouts for training."""
        obs, info = env.reset()
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(n_steps):
            # Get action
            action_mask = info['action_mask']
            action, pred_info = ppo.predict(obs, action_mask, deterministic=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # Add to buffer
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=pred_info['value'],
                log_prob=pred_info['log_prob'],
                action_mask=action_mask,
                done=terminated or truncated
            )
            
            # Track episode stats
            current_episode_reward += reward
            current_episode_length += 1
            
            if terminated or truncated:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                # Calculate success (partial completion is acceptable)
                completion_rate = next_info['tasks_scheduled'] / next_info['total_tasks']
                success = completion_rate >= 0.8  # 80% completion counts as success
                episode_success.append(float(success))
                
                current_episode_reward = 0
                current_episode_length = 0
                
                # Reset environment
                obs, info = env.reset()
            else:
                obs = next_obs
                info = next_info
                
        # Get last value for bootstrapping
        if not (terminated or truncated):
            action_mask = info['action_mask']
            _, pred_info = ppo.predict(obs, action_mask, deterministic=False)
            last_value = pred_info['value']
        else:
            last_value = 0.0
            
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            last_value=last_value
        )
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_success': episode_success,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'n_episodes': len(episode_rewards)
        }
        
    def train_curriculum(self) -> Dict:
        """
        Train through all curriculum stages.
        
        Returns:
            Training results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Curriculum Training with {len(self.stages)} stages")
        logger.info(f"{'='*60}\n")
        
        ppo = None
        
        for idx, stage in enumerate(self.stages):
            # Train stage
            ppo, metrics = self.train_stage(stage, ppo, idx)
            self.stage_results.append(metrics)
            
            # Log stage completion
            self.writer.add_scalar('curriculum/stage_completed', idx + 1, self.global_timesteps)
            
            # Check if ready for next stage
            if metrics['final_success_rate'] < stage.success_threshold:
                logger.warning(f"Stage {idx + 1} did not meet success threshold")
                logger.warning(f"Achieved: {metrics['final_success_rate']:.1%}, Required: {stage.success_threshold:.1%}")
                if idx < len(self.stages) - 1:
                    logger.info("Continuing to next stage anyway...")
                    
        # Save final model
        if ppo:
            final_path = os.path.join(self.checkpoint_dir, "curriculum_final.pth")
            ppo.save(final_path)
            logger.info(f"Saved final curriculum model to {final_path}")
            
        # Save results
        results = {
            'stages': self.stage_results,
            'total_timesteps': self.global_timesteps,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        }
        
        results_path = os.path.join(self.checkpoint_dir, "curriculum_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"\n{'='*60}")
        logger.info("Curriculum Training Complete!")
        logger.info(f"Total timesteps: {self.global_timesteps}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"{'='*60}\n")
        
        # Print summary
        self._print_summary()
        
        return results
        
    def _print_summary(self):
        """Print training summary."""
        print("\nCurriculum Training Summary:")
        print("-" * 40)
        
        for idx, (stage, result) in enumerate(zip(self.stages, self.stage_results)):
            print(f"\nStage {idx + 1}: {stage.name}")
            print(f"  Data: {os.path.basename(stage.data_path)}")
            print(f"  Episodes: {result['episodes']}")
            print(f"  Best reward: {result['best_reward']:.1f}")
            print(f"  Success rate: {result['best_success_rate']:.1%}")
            print(f"  Final success: {result['final_success_rate']:.1%}")
            
        print("\n" + "=" * 40)


def get_default_stages() -> List[StageConfig]:
    """Get default curriculum stages with adjusted thresholds."""
    return [
        StageConfig(
            name="toy_easy",
            data_path="data/10_jobs.json",
            n_timesteps=10000,  # Reduced for faster training
            success_threshold=0.7,  # More realistic threshold
            min_reward=0,
            description="Learn basic sequencing with 10 jobs"
        ),
        StageConfig(
            name="toy_normal",
            data_path="data/20_jobs.json",
            n_timesteps=20000,  # Reduced for faster training
            success_threshold=0.6,  # Lower threshold
            min_reward=0,
            description="Handle urgency with 20 jobs"
        ),
        StageConfig(
            name="small",
            data_path="data/40_jobs.json",
            n_timesteps=30000,  # Reduced for faster training
            success_threshold=0.5,  # Lower threshold
            min_reward=0,
            description="Resource contention with 40 jobs"
        ),
        StageConfig(
            name="medium",
            data_path="data/60_jobs.json",
            n_timesteps=40000,  # Reduced for faster training
            success_threshold=0.4,  # Lower threshold
            min_reward=0,
            description="Complex dependencies with 60 jobs"
        ),
        StageConfig(
            name="large",
            data_path="data/100_jobs.json",
            n_timesteps=50000,  # Reduced for faster training
            success_threshold=0.3,  # Lower threshold
            min_reward=0,
            description="Near production scale with 100 jobs"
        ),
        StageConfig(
            name="production",
            data_path="data/200_jobs.json",
            n_timesteps=60000,  # Reduced for faster training
            success_threshold=0.2,  # Lower threshold
            min_reward=0,
            description="Full production complexity with 200+ jobs"
        )
    ]


def main():
    """Main entry point for curriculum training."""
    parser = argparse.ArgumentParser(description="Curriculum training for PPO scheduling")
    parser.add_argument(
        "--stages",
        type=str,
        nargs='+',
        default=None,
        help="Stage data files to use"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/curriculum",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="tensorboard/curriculum",
        help="Tensorboard directory"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per rollout"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    # Create PPO config
    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        device=args.device
    )
    
    # Get stages
    if args.stages:
        stages = [
            StageConfig(
                name=f"stage_{i}",
                data_path=path,
                n_timesteps=100000,
                success_threshold=0.7
            )
            for i, path in enumerate(args.stages)
        ]
    else:
        stages = get_default_stages()
        
    # Create trainer
    trainer = CurriculumTrainer(
        stages=stages,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
        config=config
    )
    
    # Run curriculum
    results = trainer.train_curriculum()
    
    return results


if __name__ == "__main__":
    main()