"""
Final training script with fixed environment and clean data
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from phase3.environments.curriculum_env_fixed import CurriculumEnvironmentFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMonitor(BaseCallback):
    """Monitor training progress."""
    def __init__(self, stage_name: str):
        super().__init__()
        self.stage_name = stage_name
        self.episode_count = 0
        self.episode_rewards = []
        self.scheduling_rates = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            reward = info.get('episode', {}).get('r', 0)
            self.episode_rewards.append(reward)
            self.episode_count += 1
            
            # Get scheduling rate every 100 episodes
            if self.episode_count % 100 == 0:
                env = self.training_env.envs[0].env
                scheduled = len(env.scheduled_jobs)
                total = env.total_tasks
                rate = scheduled / total if total > 0 else 0
                self.scheduling_rates.append(rate)
                
                logger.info(f"{self.stage_name} | Episode {self.episode_count} | "
                          f"Avg Reward: {np.mean(self.episode_rewards[-100:]):.1f} | "
                          f"Schedule Rate: {rate:.1%}")
        return True


def train_stage(stage_name: str, timesteps: int = 100000) -> dict:
    """Train a single stage."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name}")
    logger.info(f"{'='*60}")
    
    # Create environment
    def make_env():
        env = CurriculumEnvironmentFixed(stage_name, verbose=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    # Note: NOT using VecNormalize to avoid reward normalization issues
    
    # Create model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,  # Smaller for more frequent updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.05,  # Some exploration but not too much
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        verbose=0
    )
    
    # Training
    monitor = TrainingMonitor(stage_name)
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps, callback=monitor)
    
    training_time = time.time() - start_time
    
    # Evaluation
    logger.info(f"\nEvaluating {stage_name}...")
    eval_episodes = 20
    total_scheduled = 0
    total_possible = 0
    eval_rewards = []
    
    for ep in range(eval_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1
        
        eval_rewards.append(ep_reward)
        
        # Get final stats
        base_env = env.envs[0].env
        scheduled = len(base_env.scheduled_jobs)
        total = base_env.total_tasks
        total_scheduled += scheduled
        total_possible += total
        
        if ep < 5:  # Show first few episodes
            logger.info(f"  Episode {ep+1}: Scheduled {scheduled}/{total} "
                      f"({scheduled/total*100:.1f}%), Reward: {float(ep_reward):.1f}")
    
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/final_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_model.zip")
    model.save(model_path)
    
    results = {
        'stage': stage_name,
        'scheduling_rate': final_rate,
        'avg_reward': np.mean(eval_rewards),
        'training_time_min': training_time / 60,
        'total_episodes': monitor.episode_count,
        'model_path': model_path
    }
    
    logger.info(f"\n{stage_name} Final Results:")
    logger.info(f"  Scheduling rate: {final_rate:.1%}")
    logger.info(f"  Average reward: {results['avg_reward']:.1f}")
    logger.info(f"  Training time: {results['training_time_min']:.1f} minutes")
    
    env.close()
    return results


def main():
    """Train all 4 toy stages."""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    logger.info("="*60)
    logger.info("FINAL TRAINING - FIXED ENVIRONMENT WITH CLEAN DATA")
    logger.info("="*60)
    logger.info("Improvements:")
    logger.info("  - Proper no-action support")
    logger.info("  - Clean data (only schedulable tasks)")
    logger.info("  - Fixed reward structure")
    logger.info("="*60)
    
    all_results = {}
    
    for stage in stages:
        results = train_stage(stage)
        all_results[stage] = results
        time.sleep(2)  # Brief pause between stages
    
    # Save summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'stages': all_results,
        'overall_metrics': {
            'avg_scheduling_rate': np.mean([r['scheduling_rate'] for r in all_results.values()]),
            'total_training_time_min': sum(r['training_time_min'] for r in all_results.values())
        }
    }
    
    summary_path = "/Users/carrickcheah/Project/ppo/app_2/phase3/results/final_training_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info("\nFinal Scheduling Rates:")
    for stage, results in all_results.items():
        logger.info(f"  {stage}: {results['scheduling_rate']:.1%}")
    logger.info(f"\nAverage: {summary['overall_metrics']['avg_scheduling_rate']:.1%}")
    logger.info(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()