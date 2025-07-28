"""
Train all toy stages to 100% performance
Implements adaptive training with performance monitoring
"""

import os
import sys
import time
import json
import logging
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor(BaseCallback):
    """Monitor performance and adapt training if needed."""
    
    def __init__(self, stage_name: str, target_rate: float = 1.0, check_freq: int = 10000):
        super().__init__()
        self.stage_name = stage_name
        self.target_rate = target_rate
        self.check_freq = check_freq
        self.best_rate = 0
        self.episodes_without_improvement = 0
        self.performance_history = []
        self.achieved_target = False
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate current performance
            env = self.training_env.envs[0].env
            
            # Run 5 evaluation episodes
            total_scheduled = 0
            total_possible = 0
            
            for _ in range(5):
                obs = env.reset()[0]
                done = False
                steps = 0
                
                while not done and steps < 200:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(action)
                    steps += 1
                
                scheduled = len(env.scheduled_jobs) if hasattr(env, 'scheduled_jobs') else 0
                total = env.total_tasks if hasattr(env, 'total_tasks') else 1
                total_scheduled += scheduled
                total_possible += total
            
            current_rate = total_scheduled / total_possible if total_possible > 0 else 0
            self.performance_history.append((self.n_calls, current_rate))
            
            logger.info(f"[{self.stage_name}] Step {self.n_calls}: Performance {current_rate:.1%}")
            
            # Check if we've achieved target
            if current_rate >= self.target_rate:
                logger.info(f"[{self.stage_name}] TARGET ACHIEVED! {current_rate:.1%} >= {self.target_rate:.1%}")
                self.achieved_target = True
                return False  # Stop training
            
            # Track improvement
            if current_rate > self.best_rate:
                self.best_rate = current_rate
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            
            # Adapt if stuck
            if self.episodes_without_improvement >= 5 and current_rate < 0.5:
                logger.warning(f"[{self.stage_name}] Stuck at {current_rate:.1%}, adjusting hyperparameters")
                # Increase exploration
                self.model.ent_coef = min(0.2, self.model.ent_coef * 1.5)
                logger.info(f"[{self.stage_name}] Increased entropy to {self.model.ent_coef}")
        
        return True


def create_improved_model(env, stage_name: str):
    """Create PPO model with improved hyperparameters for 100% performance."""
    
    # Stage-specific configurations
    stage_configs = {
        'toy_easy': {
            'learning_rate': 3e-4,
            'ent_coef': 0.05,  # Less exploration needed
            'n_steps': 512,
            'batch_size': 64,
        },
        'toy_normal': {
            'learning_rate': 5e-4,  # Higher LR for faster learning
            'ent_coef': 0.1,       # More exploration for deadlines
            'n_steps': 1024,       # More steps per update
            'batch_size': 128,
        },
        'toy_hard': {
            'learning_rate': 5e-4,
            'ent_coef': 0.15,      # Even more exploration for priorities
            'n_steps': 1024,
            'batch_size': 128,
        },
        'toy_multi': {
            'learning_rate': 5e-4,
            'ent_coef': 0.1,
            'n_steps': 1024,
            'batch_size': 128,
        }
    }
    
    config = stage_configs.get(stage_name, stage_configs['toy_normal'])
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=config['ent_coef'],
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network
        ),
        verbose=1
    )
    
    return model


def train_to_perfection(stage_name: str, max_timesteps: int = 1000000):
    """Train a stage until it achieves 100% performance."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name} to 100% performance")
    logger.info(f"{'='*60}")
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create model with improved hyperparameters
    model = create_improved_model(env, stage_name)
    
    # Setup logging
    log_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/logs/{stage_name}_perfect"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Callbacks
    checkpoint_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/perfect/{stage_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    performance_monitor = PerformanceMonitor(stage_name, target_rate=1.0, check_freq=10000)
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=checkpoint_dir,
        name_prefix=f"{stage_name}_checkpoint"
    )
    
    callbacks = [performance_monitor, checkpoint_callback]
    
    # Training
    start_time = time.time()
    logger.info(f"Training up to {max_timesteps} timesteps or until 100% achieved...")
    
    try:
        model.learn(total_timesteps=max_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    logger.info(f"\nFinal evaluation for {stage_name}...")
    
    total_scheduled = 0
    total_possible = 0
    rewards = []
    
    for ep in range(20):  # 20 episodes for robust evaluation
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
            steps += 1
        
        # Get actual environment from vec env
        actual_env = env.envs[0].env
        scheduled = len(actual_env.scheduled_jobs) if hasattr(actual_env, 'scheduled_jobs') else 0
        total = actual_env.total_tasks if hasattr(actual_env, 'total_tasks') else 1
        
        total_scheduled += scheduled
        total_possible += total
        rewards.append(ep_reward)
    
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/perfect_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_100percent.zip")
    model.save(model_path)
    
    # Save results
    results = {
        'stage': stage_name,
        'final_rate': final_rate,
        'average_reward': np.mean(rewards),
        'training_time_min': training_time / 60,
        'total_timesteps': performance_monitor.n_calls,
        'achieved_target': performance_monitor.achieved_target,
        'performance_history': performance_monitor.performance_history,
        'model_path': model_path
    }
    
    results_path = os.path.join(save_dir, f"{stage_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{stage_name} Results:")
    logger.info(f"  Final performance: {final_rate:.1%}")
    logger.info(f"  Average reward: {np.mean(rewards):.1f}")
    logger.info(f"  Training time: {training_time/60:.1f} minutes")
    logger.info(f"  Model saved to: {model_path}")
    
    return results


def main():
    """Train all toy stages to 100% performance."""
    
    # Check if we have Apple Silicon
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) detected and will be used for training")
    
    stages = ['toy_normal', 'toy_hard', 'toy_multi']  # toy_easy already at 100%
    
    # Different max timesteps based on complexity
    timesteps_config = {
        'toy_normal': 500000,   # 5x more than before
        'toy_hard': 500000,
        'toy_multi': 750000     # Multi-machine is harder
    }
    
    all_results = {}
    
    for stage in stages:
        max_timesteps = timesteps_config.get(stage, 500000)
        results = train_to_perfection(stage, max_timesteps)
        all_results[stage] = results
        
        if results['final_rate'] < 1.0:
            logger.warning(f"{stage} did not achieve 100% ({results['final_rate']:.1%})")
            logger.info("Consider increasing timesteps or adjusting reward structure further")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY - Target: 100% for all stages")
    print("="*60)
    
    # Include toy_easy which is already at 100%
    print(f"{'Stage':<15} {'Performance':<15} {'Status':<20}")
    print("-"*50)
    print(f"{'toy_easy':<15} {'100.0%':<15} {'✓ Already Perfect':<20}")
    
    for stage, results in all_results.items():
        perf = f"{results['final_rate']:.1%}"
        status = "✓ Perfect!" if results['final_rate'] >= 1.0 else "✗ Needs More Training"
        print(f"{stage:<15} {perf:<15} {status:<20}")
    
    # Check if all achieved 100%
    all_perfect = all(r['final_rate'] >= 1.0 for r in all_results.values())
    
    if all_perfect:
        print("\n✓ ALL TOY STAGES ACHIEVED 100% PERFORMANCE!")
        print("Ready to move to the next phase!")
    else:
        print("\n✗ Some stages need more training to reach 100%")
        print("Run training again with more timesteps or adjusted parameters")


if __name__ == "__main__":
    main()