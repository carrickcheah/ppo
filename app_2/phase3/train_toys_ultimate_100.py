"""Ultimate training script to achieve 100% toy performance
Completely overrides rewards to focus ONLY on scheduling"""

import os
import sys
import time
import json
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltimateCompletionWrapper(Monitor):
    """Ultimate wrapper - MASSIVE rewards for any scheduling"""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_scheduled = 0
        self.total_steps = 0
        
    def reset(self, **kwargs):
        self.episode_scheduled = 0
        self.total_steps = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, original_reward, done, truncated, info = super().step(action)
        self.total_steps += 1
        
        # Complete reward override
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            # HUGE reward for scheduling
            reward = 1000.0
            self.episode_scheduled += 1
            logger.info(f"Scheduled job {self.episode_scheduled}! Action: {action}")
        elif info.get('action_valid', False):
            # Small positive for valid wait
            reward = 1.0
        else:
            # Very small penalty for invalid
            reward = -0.1
        
        # Massive episode completion bonus
        if done or truncated:
            total_tasks = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            completion_rate = self.episode_scheduled / total_tasks if total_tasks > 0 else 0
            
            # Exponential bonuses
            if completion_rate >= 1.0:
                reward += 10000.0  # MASSIVE bonus
                logger.info(f"100% COMPLETION! {self.episode_scheduled}/{total_tasks}")
            elif completion_rate >= 0.9:
                reward += 5000.0
            elif completion_rate >= 0.8:
                reward += 2000.0
            elif completion_rate >= 0.7:
                reward += 1000.0
            elif completion_rate >= 0.6:
                reward += 500.0
            
            logger.info(f"Episode end: {self.episode_scheduled}/{total_tasks} ({completion_rate:.1%}), steps: {self.total_steps}")
        
        return obs, reward, done, truncated, info


class ProgressMonitor(BaseCallback):
    """Monitor training progress"""
    
    def __init__(self, check_freq: int = 1000, stage_name: str = ""):
        super().__init__()
        self.check_freq = check_freq
        self.stage_name = stage_name
        self.best_rate = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            logger.info(f"[{self.stage_name}] Step {self.n_calls}")
        return True


def ultimate_train(stage_name: str, timesteps: int = 1000000):
    """Ultimate training for 100% completion"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ULTIMATE TRAINING: {stage_name}")
    logger.info(f"Target: 100% completion NO MATTER WHAT")
    logger.info(f"{'='*60}")
    
    # Create wrapped environment
    base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = UltimateCompletionWrapper(base_env)
    env = DummyVecEnv([lambda: env])
    
    # Aggressive hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-3,  # High learning rate
        n_steps=512,  # Smaller batches for faster updates
        batch_size=64,
        n_epochs=20,  # More epochs per update
        gamma=0.95,  # Less future discounting
        gae_lambda=0.9,
        clip_range=0.3,  # Larger clip range
        vf_coef=0.5,
        ent_coef=0.5,  # VERY high exploration
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512], vf=[512, 512])  # Bigger network
        ),
        verbose=1
    )
    
    # Logging
    log_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/logs/{stage_name}_ultimate"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Training
    start_time = time.time()
    callback = ProgressMonitor(check_freq=5000, stage_name=stage_name)
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback, log_interval=5)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    
    # Evaluation
    logger.info(f"\nEvaluating {stage_name}...")
    eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    results = []
    for ep in range(10):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:  # More steps allowed
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            steps += 1
            done = done or truncated
        
        scheduled = len(eval_env.scheduled_jobs) if hasattr(eval_env, 'scheduled_jobs') else 0
        total = eval_env.total_tasks if hasattr(eval_env, 'total_tasks') else 1
        rate = scheduled / total if total > 0 else 0
        results.append(rate)
        
        if ep == 0:
            logger.info(f"First episode: {scheduled}/{total} = {rate:.1%}")
    
    avg_rate = np.mean(results)
    
    # Save if good
    if avg_rate >= 0.9:
        save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/ultimate_models"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{stage_name}_ultimate.zip")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    return {
        'stage': stage_name,
        'avg_rate': avg_rate,
        'training_time': time.time() - start_time
    }


def main():
    """Train all stages needing improvement"""
    
    stages = {
        'toy_normal': 1000000,  # 1M timesteps
        'toy_hard': 1000000,
        'toy_multi': 1000000
    }
    
    results = {}
    
    for stage, timesteps in stages.items():
        logger.info(f"\nTraining {stage} for {timesteps} timesteps...")
        result = ultimate_train(stage, timesteps)
        results[stage] = result
        
        if result['avg_rate'] >= 1.0:
            logger.info(f"✓ {stage} ACHIEVED 100%!")
        else:
            logger.info(f"✗ {stage} achieved {result['avg_rate']:.1%}")
    
    # Summary
    print("\n" + "="*60)
    print("ULTIMATE TRAINING RESULTS")
    print("="*60)
    
    all_perfect = True
    for stage, result in results.items():
        rate = result['avg_rate']
        status = "✓ PERFECT!" if rate >= 1.0 else f"✗ {rate:.1%}"
        print(f"{stage}: {status}")
        if rate < 1.0:
            all_perfect = False
    
    if all_perfect:
        print("\n✓ ALL TOYS ACHIEVED 100% PERFORMANCE!")
        print("Ready to move to the next phase!")
    else:
        print("\n✗ Some stages still need work")
        print("Consider even more aggressive rewards or longer training")


if __name__ == "__main__":
    main()