"""
Train with FIXED environment that has proper no-action
"""

import os
import sys
import yaml
import time
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from phase3.environments.curriculum_env_fixed import CurriculumEnvironmentFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_fixed_foundation():
    """Train with the FIXED environment."""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    # Output directory
    checkpoint_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/fixed_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TRAINING WITH FIXED ENVIRONMENT")
    logger.info("="*60)
    logger.info("Key fixes:")
    logger.info("  - Real no-action support")
    logger.info("  - Proper reward structure")
    logger.info("  - No free rewards")
    logger.info("="*60)
    
    results = {}
    
    for stage in stages:
        logger.info(f"\nTraining {stage}...")
        
        # Create environment
        def make_env():
            env = CurriculumEnvironmentFixed(stage, verbose=False)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Model parameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,  # Lower entropy for focused learning
            verbose=1,
            tensorboard_log=f"./tb_logs/{stage}_fixed"
        )
        
        # Train
        start_time = time.time()
        timesteps = 50000  # Same as before
        
        model.learn(total_timesteps=timesteps)
        
        # Evaluate
        logger.info(f"\nEvaluating {stage}...")
        total_scheduled = 0
        total_possible = 0
        rewards = []
        
        for _ in range(10):
            obs = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
            
            # Get stats from underlying env
            base_env = env.envs[0].env
            total_scheduled += len(base_env.scheduled_jobs)
            total_possible += base_env.total_tasks
            rewards.append(ep_reward)
        
        scheduling_rate = total_scheduled / total_possible if total_possible > 0 else 0
        avg_reward = sum(rewards) / len(rewards)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, f"{stage}_fixed_model.zip")
        model.save(model_path)
        
        results[stage] = {
            'scheduling_rate': scheduling_rate,
            'avg_reward': avg_reward,
            'training_time': time.time() - start_time
        }
        
        logger.info(f"\n{stage} Results:")
        logger.info(f"  Scheduling rate: {scheduling_rate:.1%}")
        logger.info(f"  Average reward: {avg_reward:.1f}")
        
        env.close()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - FIXED ENVIRONMENT")
    logger.info("="*60)
    logger.info("\nResults:")
    for stage, res in results.items():
        logger.info(f"  {stage}: {res['scheduling_rate']:.1%} scheduling rate")
    
    return results


if __name__ == "__main__":
    train_fixed_foundation()