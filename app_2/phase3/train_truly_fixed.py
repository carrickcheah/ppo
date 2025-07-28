"""
Final training with TRULY FIXED environment - no free rewards!
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

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetailedTrainingMonitor(BaseCallback):
    """Monitor with detailed tracking."""
    def __init__(self, stage_name: str, log_freq: int = 50):
        super().__init__()
        self.stage_name = stage_name
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.scheduling_rates = []
        self.actions_taken = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            reward = info.get('episode', {}).get('r', 0)
            self.episode_rewards.append(reward)
            self.episode_count += 1
            
            # Get detailed stats
            env = self.training_env.envs[0].env
            scheduled = len(env.scheduled_jobs)
            total = env.total_tasks
            rate = scheduled / total if total > 0 else 0
            self.scheduling_rates.append(rate)
            
            # Log every N episodes
            if self.episode_count % self.log_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_freq:])
                avg_rate = np.mean(self.scheduling_rates[-self.log_freq:])
                
                logger.info(f"{self.stage_name} | Episode {self.episode_count} | "
                          f"Avg Reward: {avg_reward:.1f} | "
                          f"Avg Schedule Rate: {avg_rate:.1%}")
                
                # Log if improving
                if avg_rate > 0.5:
                    logger.info(f"  -> GOOD PROGRESS! Scheduling rate above 50%")
                    
        return True


def train_stage_truly_fixed(stage_name: str, timesteps: int = 200000) -> dict:
    """Train a single stage with truly fixed rewards."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name} - TRULY FIXED REWARDS")
    logger.info(f"{'='*60}")
    
    # Create environment
    def make_env():
        env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create model with exploration-encouraging hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,      # Higher for faster learning
        n_steps=256,             # Smaller batches
        batch_size=32,           # Smaller batch size
        n_epochs=10,
        gamma=0.95,              # Less future-focused
        gae_lambda=0.9,
        clip_range=0.3,          # More exploration
        vf_coef=0.5,
        ent_coef=0.1,            # Higher entropy for exploration
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Larger network
            activation_fn=torch.nn.ReLU
        ),
        verbose=0
    )
    
    # Training
    monitor = DetailedTrainingMonitor(stage_name)
    start_time = time.time()
    
    logger.info(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=monitor)
    
    training_time = time.time() - start_time
    
    # Detailed evaluation
    logger.info(f"\nEvaluating {stage_name}...")
    eval_episodes = 50
    total_scheduled = 0
    total_possible = 0
    eval_rewards = []
    eval_details = []
    
    for ep in range(eval_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        actions_in_episode = {'schedule': 0, 'no_action': 0, 'invalid': 0}
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            
            # Track action types
            action_type = info[0].get('action_type', 'unknown')
            if action_type in actions_in_episode:
                actions_in_episode[action_type] += 1
            
            steps += 1
        
        eval_rewards.append(ep_reward)
        
        # Get final stats
        base_env = env.envs[0].env
        scheduled = len(base_env.scheduled_jobs)
        total = base_env.total_tasks
        total_scheduled += scheduled
        total_possible += total
        
        eval_details.append({
            'scheduled': scheduled,
            'total': total,
            'rate': scheduled/total if total > 0 else 0,
            'reward': float(ep_reward),
            'actions': actions_in_episode
        })
        
        if ep < 5:  # Show first few
            logger.info(f"  Episode {ep+1}: Scheduled {scheduled}/{total} "
                      f"({scheduled/total*100:.1f}%), Reward: {float(ep_reward):.1f}, "
                      f"Actions: {actions_in_episode}")
    
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_model.zip")
    model.save(model_path)
    
    # Calculate action distribution
    total_actions = sum(sum(d['actions'].values()) for d in eval_details)
    schedule_actions = sum(d['actions']['schedule'] for d in eval_details)
    
    results = {
        'stage': stage_name,
        'scheduling_rate': final_rate,
        'avg_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'min_rate': min(d['rate'] for d in eval_details),
        'max_rate': max(d['rate'] for d in eval_details),
        'schedule_action_rate': schedule_actions / total_actions if total_actions > 0 else 0,
        'training_time_min': training_time / 60,
        'total_episodes': monitor.episode_count,
        'model_path': model_path
    }
    
    logger.info(f"\n{stage_name} Final Results:")
    logger.info(f"  Scheduling rate: {final_rate:.1%} (min: {results['min_rate']:.1%}, max: {results['max_rate']:.1%})")
    logger.info(f"  Average reward: {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
    logger.info(f"  Schedule action rate: {results['schedule_action_rate']:.1%}")
    logger.info(f"  Training time: {results['training_time_min']:.1f} minutes")
    
    env.close()
    return results


def main():
    """Train all 4 toy stages with truly fixed rewards."""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    logger.info("="*60)
    logger.info("TRULY FIXED TRAINING - NO FREE REWARDS")
    logger.info("="*60)
    logger.info("Key fixes:")
    logger.info("  - NO bonus rewards at episode end")
    logger.info("  - Agent must actively schedule to get rewards")
    logger.info("  - Proper no-action support")
    logger.info("  - Clean data (only schedulable tasks)")
    logger.info("="*60)
    
    all_results = {}
    
    for stage in stages:
        results = train_stage_truly_fixed(stage)
        all_results[stage] = results
        
        # Early stopping if getting good results
        if results['scheduling_rate'] > 0.8:
            logger.info(f"\nEXCELLENT! {stage} achieved {results['scheduling_rate']:.1%} scheduling rate!")
        
        time.sleep(2)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'training_date': datetime.now().isoformat(),
        'stages': all_results,
        'overall_metrics': {
            'avg_scheduling_rate': np.mean([r['scheduling_rate'] for r in all_results.values()]),
            'min_scheduling_rate': min(r['scheduling_rate'] for r in all_results.values()),
            'max_scheduling_rate': max(r['scheduling_rate'] for r in all_results.values()),
            'total_training_time_min': sum(r['training_time_min'] for r in all_results.values())
        }
    }
    
    summary_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/results/truly_fixed_training_{timestamp}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - TRULY FIXED")
    logger.info("="*60)
    logger.info("\nFinal Scheduling Rates:")
    for stage, results in all_results.items():
        status = "SUCCESS" if results['scheduling_rate'] > 0.5 else "NEEDS WORK"
        logger.info(f"  {stage}: {results['scheduling_rate']:.1%} - {status}")
    
    logger.info(f"\nOverall:")
    logger.info(f"  Average: {summary['overall_metrics']['avg_scheduling_rate']:.1%}")
    logger.info(f"  Range: {summary['overall_metrics']['min_scheduling_rate']:.1%} - {summary['overall_metrics']['max_scheduling_rate']:.1%}")
    logger.info(f"\nResults saved to: {summary_path}")
    
    # Success check
    if summary['overall_metrics']['avg_scheduling_rate'] > 0.5:
        logger.info("\n✓ SUCCESS! Average scheduling rate above 50%!")
    else:
        logger.info("\n✗ More tuning needed - scheduling rate still low")


if __name__ == "__main__":
    main()