"""
Train all 4 toy stages with truly fixed environment
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressMonitor(BaseCallback):
    """Monitor training progress with early stopping."""
    def __init__(self, stage_name: str, target_rate: float = 0.8):
        super().__init__()
        self.stage_name = stage_name
        self.target_rate = target_rate
        self.episode_count = 0
        self.best_rate = 0
        self.recent_rates = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            # Check performance every 100 episodes
            if self.episode_count % 100 == 0:
                env = self.training_env.envs[0].env
                
                # Quick evaluation
                obs = env.reset()[0]
                done = False
                steps = 0
                
                while not done and steps < 100:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(action)
                    steps += 1
                
                scheduled = len(env.scheduled_jobs)
                total = env.total_tasks
                rate = scheduled / total if total > 0 else 0
                
                self.recent_rates.append(rate)
                if rate > self.best_rate:
                    self.best_rate = rate
                
                avg_rate = np.mean(self.recent_rates[-10:]) if self.recent_rates else 0
                
                logger.info(f"{self.stage_name} | Episode {self.episode_count} | "
                          f"Current: {rate:.1%} | Best: {self.best_rate:.1%} | "
                          f"Avg(10): {avg_rate:.1%}")
                
                # Early stopping if target reached
                if avg_rate >= self.target_rate:
                    logger.info(f"TARGET REACHED! Stopping training.")
                    return False
                    
        return True


def train_toy_stage(stage_name: str, max_timesteps: int = 100000) -> dict:
    """Train a single toy stage with optimized settings."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name}")
    logger.info(f"{'='*60}")
    
    # Create environment
    def make_env():
        env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Optimized hyperparameters for faster learning
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,       # Higher for faster initial learning
        n_steps=128,              # Smaller batches for quicker updates
        batch_size=32,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.05,            # Some exploration
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        ),
        verbose=0
    )
    
    # Callbacks
    checkpoint_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/truly_fixed/{stage_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ProgressMonitor(stage_name, target_rate=0.8),
        CheckpointCallback(
            save_freq=10000,
            save_path=checkpoint_dir,
            name_prefix=f"{stage_name}_checkpoint"
        )
    ]
    
    # Training
    start_time = time.time()
    logger.info(f"Training up to {max_timesteps} timesteps...")
    
    try:
        model.learn(total_timesteps=max_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Detailed evaluation
    logger.info(f"\nEvaluating {stage_name}...")
    eval_episodes = 30
    results = evaluate_model(model, env, eval_episodes)
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_final.zip")
    model.save(model_path)
    
    results.update({
        'stage': stage_name,
        'training_time_min': training_time / 60,
        'model_path': model_path
    })
    
    # Print results
    logger.info(f"\n{stage_name} Results:")
    logger.info(f"  Scheduling rate: {results['scheduling_rate']:.1%}")
    logger.info(f"  Average reward: {results['avg_reward']:.1f}")
    logger.info(f"  Training time: {results['training_time_min']:.1f} minutes")
    
    env.close()
    return results


def evaluate_model(model, env, n_episodes: int = 30) -> dict:
    """Evaluate model performance."""
    total_scheduled = 0
    total_possible = 0
    episode_rewards = []
    scheduling_rates = []
    action_distributions = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        actions = {'schedule': 0, 'no_action': 0, 'invalid': 0}
        
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            
            action_type = info[0].get('action_type', 'unknown')
            if action_type in actions:
                actions[action_type] += 1
            
            steps += 1
        
        # Get stats from base env
        base_env = env.envs[0].env
        scheduled = len(base_env.scheduled_jobs)
        total = base_env.total_tasks
        
        total_scheduled += scheduled
        total_possible += total
        episode_rewards.append(ep_reward)
        
        rate = scheduled / total if total > 0 else 0
        scheduling_rates.append(rate)
        action_distributions.append(actions)
    
    return {
        'scheduling_rate': total_scheduled / total_possible if total_possible > 0 else 0,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_rate': min(scheduling_rates),
        'max_rate': max(scheduling_rates),
        'avg_schedule_actions': np.mean([a['schedule'] for a in action_distributions]),
        'avg_invalid_actions': np.mean([a['invalid'] for a in action_distributions])
    }


def main():
    """Train all 4 toy stages."""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    logger.info("="*60)
    logger.info("TRAINING 4 TOY STAGES - TRULY FIXED ENVIRONMENT")
    logger.info("="*60)
    logger.info("Training order: toy_easy -> toy_normal -> toy_hard -> toy_multi")
    logger.info("Target: 80% scheduling rate for progression")
    logger.info("="*60)
    
    all_results = {}
    
    for i, stage in enumerate(stages):
        logger.info(f"\nStage {i+1}/4: {stage}")
        
        # Adjust timesteps based on difficulty
        if stage == 'toy_easy':
            max_timesteps = 50000
        elif stage == 'toy_normal':
            max_timesteps = 100000
        else:
            max_timesteps = 150000
        
        results = train_toy_stage(stage, max_timesteps)
        all_results[stage] = results
        
        # Check if we should continue
        if results['scheduling_rate'] < 0.3:
            logger.warning(f"Low performance on {stage} ({results['scheduling_rate']:.1%}). "
                         f"Consider tuning hyperparameters.")
        
        time.sleep(2)  # Brief pause
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'training_date': datetime.now().isoformat(),
        'stages': all_results,
        'overall_metrics': {
            'avg_scheduling_rate': np.mean([r['scheduling_rate'] for r in all_results.values()]),
            'stages_above_50pct': sum(1 for r in all_results.values() if r['scheduling_rate'] > 0.5),
            'stages_above_80pct': sum(1 for r in all_results.values() if r['scheduling_rate'] > 0.8),
            'total_training_time_min': sum(r['training_time_min'] for r in all_results.values())
        }
    }
    
    output_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/results/q_4toys_training_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    print("\nFinal Results:")
    print("-"*40)
    for stage, results in all_results.items():
        status = "EXCELLENT" if results['scheduling_rate'] > 0.8 else \
                 "GOOD" if results['scheduling_rate'] > 0.5 else "NEEDS WORK"
        print(f"{stage:12} | {results['scheduling_rate']:6.1%} | {status}")
    
    print(f"\nOverall Average: {summary['overall_metrics']['avg_scheduling_rate']:.1%}")
    print(f"Stages > 50%: {summary['overall_metrics']['stages_above_50pct']}/4")
    print(f"Stages > 80%: {summary['overall_metrics']['stages_above_80pct']}/4")
    print(f"\nTotal training time: {summary['overall_metrics']['total_training_time_min']:.1f} minutes")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()