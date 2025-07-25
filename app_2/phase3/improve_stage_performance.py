"""
Improve Stage Performance

Focus on training specific stages until performance improves.
"""

import os
import yaml
import logging
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.curriculum_env import CurriculumSchedulingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StageImprover:
    """Improve performance on specific stages."""
    
    def __init__(self, stage_name: str, target_reward: float = None):
        """
        Initialize stage improver.
        
        Args:
            stage_name: Stage to improve
            target_reward: Target reward to achieve (less negative is better)
        """
        self.stage_name = stage_name
        self.target_reward = target_reward
        
        # Load config
        with open('configs/phase3_curriculum_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.stage_config = self.config[stage_name]
        
        # Setup paths
        self.checkpoint_dir = f'phase3/checkpoints/{stage_name}_improved'
        self.tensorboard_dir = f'phase3/tensorboard/{stage_name}_improved'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
    def create_env(self):
        """Create environment for stage."""
        # Determine snapshot path
        snapshot_path = None
        data_source = self.stage_config['data_source']
        
        snapshot_paths = {
            'snapshot_normal': 'phase3/snapshots/snapshot_normal.json',
            'snapshot_rush': 'phase3/snapshots/snapshot_rush.json',
            'edge_same_machine': 'phase3/snapshots/edge_case_same_machine.json',
        }
        
        if 'rush' in data_source:
            snapshot_path = snapshot_paths['snapshot_rush']
        elif 'bottleneck' in data_source or 'same_machine' in data_source:
            snapshot_path = snapshot_paths['edge_same_machine']
        elif data_source != 'synthetic':
            snapshot_path = snapshot_paths['snapshot_normal']
            
        env = CurriculumSchedulingEnv(
            stage_config=self.stage_config,
            snapshot_path=snapshot_path,
            reward_profile='balanced',
            seed=42
        )
        
        return env
        
    def train_until_improved(self, max_iterations: int = 10):
        """Train stage until performance improves."""
        logger.info(f"Starting focused training for {self.stage_name}")
        logger.info(f"Target reward: {self.target_reward if self.target_reward else 'Best possible'}")
        
        # Load existing model if available
        model_path = f'phase3/checkpoints/{self.stage_name}_final.zip'
        
        best_reward = -float('inf')
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{max_iterations}")
            logger.info(f"{'='*60}")
            
            # Create fresh environments
            train_env = DummyVecEnv([self.create_env for _ in range(8)])
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            
            eval_env = DummyVecEnv([lambda: self.create_env()])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            
            # Create or load model
            if os.path.exists(model_path) and iteration == 1:
                logger.info(f"Loading existing model from {model_path}")
                model = PPO.load(model_path, env=train_env, device='cpu')  # Use CPU for stability
                # Update hyperparameters for fine-tuning
                model.learning_rate = float(self.stage_config['learning_rate']) * 0.5
            else:
                logger.info("Creating new model")
                model = PPO(
                    'MlpPolicy',
                    train_env,
                    learning_rate=float(self.stage_config['learning_rate']),
                    n_steps=int(self.stage_config['n_steps']),
                    batch_size=int(self.stage_config['batch_size']),
                    n_epochs=10,
                    gamma=0.99,
                    clip_range=0.2,
                    ent_coef=0.01,  # Encourage exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    tensorboard_log=self.tensorboard_dir,
                    device='cpu',
                    verbose=1
                )
            
            # Callbacks
            eval_callback = EvalCallback(
                eval_env,
                n_eval_episodes=5,
                eval_freq=5000,
                best_model_save_path=self.checkpoint_dir,
                deterministic=True,
                render=False
            )
            
            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path=self.checkpoint_dir,
                name_prefix=f'{self.stage_name}_iter{iteration}'
            )
            
            # Train
            logger.info(f"Training for {self.stage_config['timesteps']} timesteps")
            model.learn(
                total_timesteps=int(self.stage_config['timesteps']),
                callback=[eval_callback, checkpoint_callback],
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Evaluate final performance
            logger.info("Evaluating final performance...")
            rewards = []
            late_jobs = []
            utilizations = []
            
            for _ in range(10):
                obs = eval_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward[0]
                    
                    if done and 'episode_metrics' in info[0]:
                        metrics = info[0]['episode_metrics']
                        late_jobs.append(metrics['jobs_late'])
                        utilizations.append(metrics['machine_utilization'])
                        
                rewards.append(episode_reward)
                
            avg_reward = np.mean(rewards)
            avg_late = np.mean(late_jobs) if late_jobs else 0
            avg_util = np.mean(utilizations) if utilizations else 0
            
            logger.info(f"\nIteration {iteration} Results:")
            logger.info(f"Average Reward: {avg_reward:.2f}")
            logger.info(f"Average Late Jobs: {avg_late:.1f}")
            logger.info(f"Average Utilization: {avg_util:.1%}")
            
            # Check if improved
            if avg_reward > best_reward:
                best_reward = avg_reward
                logger.info(f"New best reward! Saving model...")
                model.save(f"{self.checkpoint_dir}/best_model.zip")
                
                # Also update the main model
                model.save(f"phase3/checkpoints/{self.stage_name}_final.zip")
                
            # Check if target reached
            if self.target_reward and avg_reward >= self.target_reward:
                logger.info(f"Target reward achieved! ({avg_reward:.2f} >= {self.target_reward})")
                break
                
            # Adjust hyperparameters for next iteration
            if iteration < max_iterations:
                logger.info("Adjusting hyperparameters for next iteration...")
                model_path = f"{self.checkpoint_dir}/best_model.zip"
                
            # Cleanup
            train_env.close()
            eval_env.close()
            
        logger.info(f"\nTraining complete! Best reward: {best_reward:.2f}")
        return best_reward


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improve stage performance')
    parser.add_argument('--stage', type=str, required=True, help='Stage to improve')
    parser.add_argument('--target', type=float, default=None, help='Target reward')
    parser.add_argument('--iterations', type=int, default=5, help='Max iterations')
    
    args = parser.parse_args()
    
    # Define target rewards for each stage
    targets = {
        'toy_easy': -5.0,
        'toy_normal': -20.0,
        'toy_hard': -30.0,
        'toy_multi': -15.0,
        'small_balanced': -100.0,
        'small_rush': -150.0,
        'small_bottleneck': -100.0
    }
    
    target = args.target or targets.get(args.stage)
    
    improver = StageImprover(args.stage, target)
    best_reward = improver.train_until_improved(args.iterations)
    
    print(f"\nFinal best reward for {args.stage}: {best_reward:.2f}")


if __name__ == '__main__':
    main()