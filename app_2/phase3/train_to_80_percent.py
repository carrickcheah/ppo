"""Train all toy stages to achieve 80% completion rate
More realistic target that balances performance with achievability"""

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


class EightyPercentWrapper(Monitor):
    """Wrapper optimized for achieving 80% completion"""
    
    def __init__(self, env, stage_name):
        super().__init__(env)
        self.stage_name = stage_name
        self.episode_scheduled = 0
        self.episode_count = 0
        
    def reset(self, **kwargs):
        self.episode_scheduled = 0
        self.episode_count += 1
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, original_reward, done, truncated, info = super().step(action)
        
        # Balanced reward structure for 80% target
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            # Good reward for scheduling, but not excessive
            reward = 200.0
            self.episode_scheduled += 1
        elif info.get('action_valid', False):
            # Small positive for valid wait
            reward = 0.5
        else:
            # Moderate penalty for invalid actions
            reward = -2.0
        
        # Episode completion bonus based on 80% target
        if done or truncated:
            total_tasks = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            completion_rate = self.episode_scheduled / total_tasks if total_tasks > 0 else 0
            
            # Bonus structure centered around 80%
            if completion_rate >= 1.0:
                reward += 2000.0  # Still reward 100%
            elif completion_rate >= 0.9:
                reward += 1500.0
            elif completion_rate >= 0.8:
                reward += 1000.0  # Target achieved!
            elif completion_rate >= 0.7:
                reward += 500.0
            elif completion_rate >= 0.6:
                reward += 200.0
            else:
                reward += 50.0 * completion_rate  # Small reward for any progress
            
            if self.episode_count % 100 == 0:
                logger.info(f"[{self.stage_name}] Episode {self.episode_count}: {completion_rate:.1%} completion")
        
        return obs, reward, done, truncated, info


class EightyPercentCallback(BaseCallback):
    """Monitor and save when 80% is achieved consistently"""
    
    def __init__(self, stage_name: str, target_rate: float = 0.8, check_freq: int = 10000):
        super().__init__()
        self.stage_name = stage_name
        self.target_rate = target_rate
        self.check_freq = check_freq
        self.best_rate = 0
        self.consecutive_good = 0
        self.save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_80_percent"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate current performance
            env = CurriculumEnvironmentTrulyFixed(self.stage_name, verbose=False)
            
            scheduled_total = 0
            possible_total = 0
            
            # Test 10 episodes
            for _ in range(10):
                obs, _ = env.reset()
                done = False
                steps = 0
                
                while not done and steps < 300:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, truncated, _ = env.step(action)
                    done = done or truncated
                    steps += 1
                
                scheduled = len(env.scheduled_jobs) if hasattr(env, 'scheduled_jobs') else 0
                total = env.total_tasks if hasattr(env, 'total_tasks') else 1
                scheduled_total += scheduled
                possible_total += total
            
            rate = scheduled_total / possible_total if possible_total > 0 else 0
            logger.info(f"[{self.stage_name}] Step {self.n_calls}: {rate:.1%} average completion")
            
            # Track consecutive good performances
            if rate >= self.target_rate:
                self.consecutive_good += 1
            else:
                self.consecutive_good = 0
            
            # Save if new best or consistently good
            if rate > self.best_rate:
                self.best_rate = rate
                if rate >= self.target_rate:
                    model_path = os.path.join(self.save_dir, f"{self.stage_name}_{int(rate*100)}.zip")
                    self.model.save(model_path)
                    logger.info(f"Saved new best model at {rate:.1%}: {model_path}")
            
            # Stop if consistently achieving target
            if self.consecutive_good >= 3:  # 3 consecutive checks at 80%+
                logger.info(f"Target achieved consistently! Stopping training.")
                return False
        
        return True


def train_to_80(stage_name: str, current_best: float, max_timesteps: int = 500000):
    """Train a single stage to 80% completion"""
    
    if current_best >= 0.8:
        logger.info(f"{stage_name} already at {current_best:.1%} - skipping")
        return current_best
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name} to 80% (current: {current_best:.1%})")
    logger.info(f"{'='*60}")
    
    # Create environment
    base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = EightyPercentWrapper(base_env, stage_name)
    env = DummyVecEnv([lambda: env])
    
    # Hyperparameters optimized for 80% target
    hyperparams = {
        'toy_normal': {
            'learning_rate': 5e-4,
            'n_steps': 2048,
            'batch_size': 128,
            'ent_coef': 0.1,  # Moderate exploration
        },
        'toy_hard': {
            'learning_rate': 1e-3,
            'n_steps': 2048,
            'batch_size': 256,
            'ent_coef': 0.15,  # More exploration
        },
        'toy_multi': {
            'learning_rate': 1e-3,
            'n_steps': 2048,
            'batch_size': 256,
            'ent_coef': 0.2,  # High exploration for complex env
        }
    }
    
    config = hyperparams.get(stage_name, hyperparams['toy_normal'])
    
    # Create model
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
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Setup logging
    log_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/logs/{stage_name}_80percent"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Callbacks
    callbacks = [
        EightyPercentCallback(stage_name, target_rate=0.8, check_freq=10000),
        CheckpointCallback(
            save_freq=50000,
            save_path=f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/80percent/{stage_name}",
            name_prefix="checkpoint"
        )
    ]
    
    # Training
    start_time = time.time()
    logger.info(f"Starting training for up to {max_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=max_timesteps,
            callback=callbacks,
            log_interval=10,
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    logger.info(f"\nFinal evaluation for {stage_name}...")
    eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    total_scheduled = 0
    total_possible = 0
    episode_rates = []
    
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward
            steps += 1
            done = done or truncated
        
        scheduled = len(eval_env.scheduled_jobs) if hasattr(eval_env, 'scheduled_jobs') else 0
        total = eval_env.total_tasks if hasattr(eval_env, 'total_tasks') else 1
        rate = scheduled / total if total > 0 else 0
        episode_rates.append(rate)
        
        total_scheduled += scheduled
        total_possible += total
        
        if ep == 0:
            logger.info(f"First episode: {scheduled}/{total} = {rate:.1%}")
    
    final_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    # Save final model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_80_percent"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_final.zip")
    model.save(model_path)
    
    # Save results
    results = {
        'stage': stage_name,
        'final_rate': final_rate,
        'average_rate': np.mean(episode_rates),
        'std_rate': np.std(episode_rates),
        'min_rate': np.min(episode_rates),
        'max_rate': np.max(episode_rates),
        'training_time_min': training_time / 60,
        'timesteps_trained': min(max_timesteps, model.num_timesteps),
        'model_path': model_path
    }
    
    results_path = os.path.join(save_dir, f"{stage_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{stage_name} Results:")
    logger.info(f"  Final rate: {final_rate:.1%}")
    logger.info(f"  Average rate: {np.mean(episode_rates):.1%} (±{np.std(episode_rates):.1%})")
    logger.info(f"  Min/Max rate: {np.min(episode_rates):.1%} / {np.max(episode_rates):.1%}")
    logger.info(f"  Training time: {training_time/60:.1f} minutes")
    
    return final_rate


def main():
    """Train all toy stages to 80% completion"""
    
    # Current best performances
    current_best = {
        'toy_easy': 1.0,     # Already perfect
        'toy_normal': 0.562, # 56.2%
        'toy_hard': 0.30,    # 30%
        'toy_multi': 0.364   # 36.4%
    }
    
    # Stages that need training
    stages_to_train = [(name, perf) for name, perf in current_best.items() 
                       if perf < 0.8 and name != 'toy_easy']
    
    logger.info("TRAINING TO 80% COMPLETION TARGET")
    logger.info("=" * 60)
    logger.info("Current status:")
    for stage, perf in current_best.items():
        status = "✓" if perf >= 0.8 else "✗"
        logger.info(f"  {stage}: {perf:.1%} {status}")
    
    # Train each stage
    final_results = current_best.copy()
    
    for stage_name, current_perf in stages_to_train:
        final_rate = train_to_80(stage_name, current_perf, max_timesteps=500000)
        final_results[stage_name] = final_rate
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS - 80% TARGET")
    print("="*60)
    print(f"{'Stage':<15} {'Performance':<15} {'Target':<10} {'Status':<20}")
    print("-"*60)
    
    all_achieved = True
    for stage in ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']:
        perf = final_results[stage]
        target = 0.8 if stage != 'toy_easy' else 1.0
        
        if perf >= target:
            status = "✓ ACHIEVED!"
        else:
            status = f"✗ {target - perf:.1%} below target"
            all_achieved = False
        
        print(f"{stage:<15} {perf:.1%}{'':14} {target:.1%}{'':9} {status:<20}")
    
    if all_achieved:
        print("\n✓ ALL TOY STAGES ACHIEVED 80% TARGET!")
        print("Ready to proceed to the next phase!")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'target': 0.8,
            'all_achieved': True,
            'results': final_results
        }
        
        with open("/Users/carrickcheah/Project/ppo/app_2/phase3/models_80_percent/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("\n✗ Some stages still below 80% target")
        print("Consider longer training or different reward structures")


if __name__ == "__main__":
    main()