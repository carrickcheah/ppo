"""Quick focused training to achieve 100% - saves as soon as good performance is reached"""

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


class QuickCompletionWrapper(Monitor):
    """Focused wrapper for quick 100% achievement"""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_scheduled = 0
        
    def reset(self, **kwargs):
        self.episode_scheduled = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        
        # Simple but effective rewards
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            reward = 500.0  # Big reward for scheduling
            self.episode_scheduled += 1
        elif info.get('action_valid', False):
            reward = 0.1  # Small positive for wait
        else:
            reward = -1.0  # Small negative for invalid
        
        # Episode completion bonus
        if done or truncated:
            total_tasks = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            completion_rate = self.episode_scheduled / total_tasks if total_tasks > 0 else 0
            
            if completion_rate >= 1.0:
                reward += 5000.0  # Big bonus for 100%
            elif completion_rate >= 0.9:
                reward += 2000.0
            elif completion_rate >= 0.8:
                reward += 1000.0
        
        return obs, reward, done, truncated, info


class EarlyStoppingCallback(BaseCallback):
    """Stop and save when performance is good"""
    
    def __init__(self, stage_name: str, check_freq: int = 5000):
        super().__init__()
        self.stage_name = stage_name
        self.check_freq = check_freq
        self.best_rate = 0
        self.save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/quick_100_models"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Quick evaluation
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):
                base_env = env.env.env if hasattr(env.env, 'env') else env.env
            else:
                base_env = env
                
            # Test a few episodes
            total_scheduled = 0
            total_possible = 0
            
            for _ in range(3):
                obs = env.reset()
                done = False
                steps = 0
                
                while not done and steps < 200:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _ = env.step(action)
                    steps += 1
                
                if hasattr(base_env, 'scheduled_jobs'):
                    scheduled = len(base_env.scheduled_jobs)
                    total = base_env.total_tasks if hasattr(base_env, 'total_tasks') else 1
                    total_scheduled += scheduled
                    total_possible += total
            
            if total_possible > 0:
                rate = total_scheduled / total_possible
                logger.info(f"[{self.stage_name}] Step {self.n_calls}: {rate:.1%} completion")
                
                # Save if good performance
                if rate > self.best_rate:
                    self.best_rate = rate
                    if rate >= 0.9:  # Save at 90%+
                        model_path = os.path.join(self.save_dir, f"{self.stage_name}_{int(rate*100)}.zip")
                        self.model.save(model_path)
                        logger.info(f"Saved model at {rate:.1%}: {model_path}")
                        
                        if rate >= 1.0:
                            logger.info("ACHIEVED 100%! Stopping training.")
                            return False  # Stop training
        
        return True


def quick_train(stage_name: str, max_timesteps: int = 200000):
    """Quick focused training"""
    
    logger.info(f"\nQuick training {stage_name} for 100%...")
    
    # Create environment
    base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = QuickCompletionWrapper(base_env)
    env = DummyVecEnv([lambda: env])
    
    # Model with good hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.3,  # Good exploration
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Setup logging
    log_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/logs/{stage_name}_quick"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Training with early stopping
    callback = EarlyStoppingCallback(stage_name, check_freq=5000)
    
    start_time = time.time()
    try:
        model.learn(total_timesteps=max_timesteps, callback=callback, log_interval=10)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    logger.info(f"\nFinal evaluation for {stage_name}...")
    eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    total_scheduled = 0
    total_possible = 0
    
    for ep in range(10):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            steps += 1
            done = done or truncated
        
        scheduled = len(eval_env.scheduled_jobs) if hasattr(eval_env, 'scheduled_jobs') else 0
        total = eval_env.total_tasks if hasattr(eval_env, 'total_tasks') else 1
        
        total_scheduled += scheduled
        total_possible += total
        
        if ep == 0:
            logger.info(f"First episode: {scheduled}/{total} = {scheduled/total:.1%}")
    
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    # Save final model
    if final_rate >= 0.5:  # Save if decent
        save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/quick_100_models"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{stage_name}_final_{int(final_rate*100)}.zip")
        model.save(model_path)
        
        # Save results
        results = {
            'stage': stage_name,
            'final_rate': final_rate,
            'best_rate': callback.best_rate,
            'training_time_min': training_time / 60,
            'model_path': model_path
        }
        
        with open(os.path.join(save_dir, f"{stage_name}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    return final_rate


def main():
    """Train all stages needing improvement"""
    
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    
    results = {'toy_easy': 1.0}  # Already perfect
    
    for stage in stages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {stage}")
        logger.info(f"{'='*60}")
        
        rate = quick_train(stage, max_timesteps=200000)
        results[stage] = rate
        
        logger.info(f"\n{stage} achieved: {rate:.1%}")
    
    # Summary
    print("\n" + "="*60)
    print("QUICK TRAINING SUMMARY")
    print("="*60)
    
    all_perfect = True
    for stage, rate in results.items():
        status = "✓ PERFECT!" if rate >= 1.0 else f"{rate:.1%}"
        print(f"{stage}: {status}")
        if rate < 1.0:
            all_perfect = False
    
    if all_perfect:
        print("\n✓ ALL TOYS ACHIEVED 100% PERFORMANCE!")
        print("We can now move to the next phase!")
    else:
        print("\n✗ Some stages still need improvement")


if __name__ == "__main__":
    main()