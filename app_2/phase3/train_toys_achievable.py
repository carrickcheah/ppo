"""
Train toy stages with achievable goals and progressive learning
Focus on scheduling completion first, then optimize
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardWrapper(Monitor):
    """Wrapper to implement progressive reward structure"""
    
    def __init__(self, env, stage_name: str, phase: str = "completion"):
        super().__init__(env)
        self.stage_name = stage_name
        self.phase = phase  # "completion" or "optimization"
        self.original_step = env.step
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # Phase 1: Focus on completion (ignore deadline penalties)
        if self.phase == "completion":
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                # Big reward for any successful scheduling
                reward = 100.0
            elif not info.get('action_valid', False):
                # Small penalty for invalid actions
                reward = -1.0
            else:
                # Neutral for wait actions
                reward = 0.0
                
            # Bonus at episode end based on completion rate
            if done:
                scheduled = len(self.env.scheduled_jobs) if hasattr(self.env, 'scheduled_jobs') else 0
                total = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
                completion_rate = scheduled / total if total > 0 else 0
                
                # Big bonus for high completion
                if completion_rate >= 1.0:
                    reward += 500.0
                elif completion_rate >= 0.8:
                    reward += 200.0
                elif completion_rate >= 0.6:
                    reward += 100.0
        
        # Phase 2: Optimize for deadlines (after achieving high completion)
        elif self.phase == "optimization":
            # Use original rewards but cap penalties
            if reward < -10:
                reward = -10  # Cap large negative rewards
            
            # Still give completion bonus
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                reward += 50.0
        
        return obs, reward, done, truncated, info


class PhaseTrainingCallback(BaseCallback):
    """Switch training phases based on performance"""
    
    def __init__(self, check_freq: int = 10000):
        super().__init__()
        self.check_freq = check_freq
        self.phase = "completion"
        self.best_completion_rate = 0
        self.phase_switched = False
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and not self.phase_switched:
            # Evaluate completion rate
            env = self.training_env.envs[0]
            
            total_scheduled = 0
            total_possible = 0
            
            for _ in range(5):
                # Handle VecEnv API
                obs = env.reset()
                dones = [False]
                steps = 0
                
                while not dones[0] and steps < 200:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)
                    steps += 1
                
                actual_env = env.env.env if hasattr(env.env, 'env') else env.env
                scheduled = len(actual_env.scheduled_jobs) if hasattr(actual_env, 'scheduled_jobs') else 0
                total = actual_env.total_tasks if hasattr(actual_env, 'total_tasks') else 1
                
                total_scheduled += scheduled
                total_possible += total
            
            completion_rate = total_scheduled / total_possible if total_possible > 0 else 0
            
            logger.info(f"Step {self.n_calls}: Completion rate = {completion_rate:.1%}")
            
            # Switch to optimization phase if completion is good
            if completion_rate >= 0.9 and self.phase == "completion":
                logger.info("SWITCHING TO OPTIMIZATION PHASE!")
                self.phase = "optimization"
                self.phase_switched = True
                
                # Update wrapper phase
                if hasattr(env.env, 'phase'):
                    env.env.phase = "optimization"
                
                # Reduce learning rate for fine-tuning
                self.model.learning_rate = self.model.learning_rate * 0.5
                
            self.best_completion_rate = max(self.best_completion_rate, completion_rate)
        
        return True


def train_with_phases(stage_name: str, max_timesteps: int = 500000):
    """Train a stage using phased approach"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name} with phased approach")
    logger.info(f"Phase 1: Focus on completion")
    logger.info(f"Phase 2: Optimize for deadlines")
    logger.info(f"{'='*60}")
    
    # Create base environment
    base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    # Wrap with reward shaping
    env = RewardWrapper(base_env, stage_name, phase="completion")
    env = DummyVecEnv([lambda: env])
    
    # Create model with high exploration initially
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-3,  # Higher initial learning rate
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.2,  # High exploration
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=1
    )
    
    # Setup logging
    log_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/logs/{stage_name}_phased"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Callbacks
    phase_callback = PhaseTrainingCallback(check_freq=10000)
    
    # Training
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=max_timesteps, callback=phase_callback)
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    
    training_time = time.time() - start_time
    
    # Final evaluation with original rewards
    logger.info(f"\nFinal evaluation for {stage_name}...")
    
    # Use original environment for final eval
    eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    total_scheduled = 0
    total_possible = 0
    rewards = []
    
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward
            steps += 1
            done = done or truncated
        
        scheduled = len(eval_env.scheduled_jobs) if hasattr(eval_env, 'scheduled_jobs') else 0
        total = eval_env.total_tasks if hasattr(eval_env, 'total_tasks') else 1
        
        total_scheduled += scheduled
        total_possible += total
        rewards.append(ep_reward)
    
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/phased_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_phased.zip")
    model.save(model_path)
    
    results = {
        'stage': stage_name,
        'final_rate': final_rate,
        'average_reward': float(np.mean(rewards)),
        'training_time_min': training_time / 60,
        'best_completion_rate': phase_callback.best_completion_rate,
        'phase_switched': phase_callback.phase_switched,
        'model_path': model_path
    }
    
    logger.info(f"\n{stage_name} Results:")
    logger.info(f"  Final performance: {final_rate:.1%}")
    logger.info(f"  Average reward: {np.mean(rewards):.1f}")
    logger.info(f"  Phase switched: {phase_callback.phase_switched}")
    logger.info(f"  Model saved to: {model_path}")
    
    return results


def main():
    """Train all toy stages that need improvement"""
    
    stages_to_train = {
        'toy_normal': 500000,
        'toy_hard': 500000, 
        'toy_multi': 500000
    }
    
    all_results = {}
    
    for stage, timesteps in stages_to_train.items():
        logger.info(f"\nTraining {stage}...")
        results = train_with_phases(stage, timesteps)
        all_results[stage] = results
        
        # Save results
        results_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/phased_models/{stage}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("PHASED TRAINING SUMMARY")
    print("="*60)
    print(f"{'Stage':<15} {'Completion':<15} {'Avg Reward':<15} {'Status':<20}")
    print("-"*65)
    
    # Include toy_easy
    print(f"{'toy_easy':<15} {'100.0%':<15} {'331.8':<15} {'✓ Already Perfect':<20}")
    
    for stage, results in all_results.items():
        perf = f"{results['final_rate']:.1%}"
        reward = f"{results['average_reward']:.1f}"
        status = "✓ Success!" if results['final_rate'] >= 0.9 else "⚠ Needs Work"
        print(f"{stage:<15} {perf:<15} {reward:<15} {status:<20}")


if __name__ == "__main__":
    main()