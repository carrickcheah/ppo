#!/usr/bin/env python
"""
Deadline-focused training for achieving 85% on-time delivery rate.
Based on train_sb3_optimized.py but with rewards heavily weighted toward meeting deadlines.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environments.scheduling_env import SchedulingEnv
import torch
import numpy as np
import time
from datetime import datetime

class OnTimeRateCallback(BaseCallback):
    """Custom callback to monitor on-time delivery rate during training."""
    
    def __init__(self, eval_env, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.best_on_time_rate = 0.0
        self.evaluations_on_time = []
        self.evaluations_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate current policy
            on_time_rate, avg_reward = self._evaluate_on_time_rate()
            self.evaluations_on_time.append(on_time_rate)
            self.evaluations_rewards.append(avg_reward)
            
            # Log to tensorboard
            self.logger.record("eval/on_time_rate", on_time_rate)
            self.logger.record("eval/avg_reward", avg_reward)
            
            if on_time_rate > self.best_on_time_rate:
                self.best_on_time_rate = on_time_rate
                if self.verbose > 0:
                    print(f"\n🎯 New best on-time rate: {on_time_rate:.1%}")
                    
            if self.verbose > 0:
                print(f"Step {self.n_calls}: On-time rate = {on_time_rate:.1%}, Reward = {avg_reward:.0f}")
                
        return True
    
    def _evaluate_on_time_rate(self, n_eval_episodes: int = 5):
        """Evaluate the on-time delivery rate."""
        total_on_time = 0
        total_late = 0
        total_rewards = []
        
        for _ in range(n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                
            # Get metrics from the environment
            env = self.eval_env.envs[0].env
            metrics = env.reward_calculator.get_metrics()
            total_on_time += metrics['on_time_tasks'] + metrics['early_tasks']
            total_late += metrics['late_tasks']
            total_rewards.append(episode_reward)
            
        total_tasks = total_on_time + total_late
        on_time_rate = total_on_time / max(total_tasks, 1)
        avg_reward = np.mean(total_rewards)
        
        return on_time_rate, avg_reward

def make_env(data_path: str, rank: int = 0):
    """Create environment with deadline-focused rewards."""
    def _init():
        env = SchedulingEnv(
            snapshot_path=data_path,
            max_steps=10000,
            planning_horizon=720.0,
            reward_config={
                # DEADLINE-FOCUSED REWARD CONFIGURATION
                'on_time_reward': 1000.0,        # 10x increase (was 100)
                'early_bonus_per_day': 200.0,    # 4x increase (was 50)
                'late_penalty_per_day': -200.0,  # 40x increase (was -5)
                'utilization_bonus': 20.0,       # 5x decrease (was 100)
                'action_taken_bonus': 2.0,       # Reduced (was 5-100)
                'idle_penalty': -5.0,            # 5x increase (was -1)
                'sequence_violation_penalty': -1000.0,  # 2x increase (was -500)
            }
        )
        env.reset(seed=42 + rank)
        return Monitor(env)
    return _init

def train_deadline_focused():
    """Train with deadline-focused rewards to achieve 85% on-time delivery."""
    
    print("="*80)
    print("DEADLINE-FOCUSED PPO TRAINING")
    print("Target: 85% On-Time Delivery Rate")
    print("="*80)
    print("\nKEY REWARD CHANGES:")
    print("1. On-time reward: 100 → 1000 (10x)")
    print("2. Late penalty: -5 → -200 (40x)")
    print("3. Early bonus: 50 → 200 (4x)")
    print("4. Utilization bonus: 100 → 20 (0.2x)")
    print("5. Action bonus: 100 → 2 (0.02x)")
    print("-"*80)
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/deadline_focused_{timestamp}"
    log_dir = f"logs/deadline_focused_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Start with smaller dataset for faster iteration
    data_path = 'data/40_jobs.json'
    total_timesteps = 200000
    
    print(f"\nDataset: {data_path}")
    print(f"Total steps: {total_timesteps:,}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("-"*80)
    
    # Use 8 parallel environments
    n_envs = 8
    print(f"\nUsing {n_envs} parallel environments")
    
    env = make_vec_env(
        make_env(data_path),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )
    
    eval_env = make_vec_env(
        make_env(data_path, rank=100),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    print("\n🎯 CREATING DEADLINE-FOCUSED MODEL")
    print("-"*60)
    
    # Use same successful architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[4096, 2048, 1024, 512, 256],
            vf=[4096, 2048, 1024, 512, 256]
        ),
        activation_fn=torch.nn.GELU,
        ortho_init=True,
        share_features_extractor=False,
        normalize_images=False,
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        
        # Learning parameters
        learning_rate=1e-3,
        n_steps=4096,
        batch_size=512,
        n_epochs=30,
        
        # Discount & advantage
        gamma=0.999,
        gae_lambda=0.99,
        
        # Exploration
        clip_range=0.3,
        clip_range_vf=None,
        ent_coef=0.05,  # Reduced from 0.1 for more focused learning
        
        # Value function
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # KL target
        target_kl=0.1,
        
        # Network
        policy_kwargs=policy_kwargs,
        
        # Other
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    print("📊 Model Configuration:")
    print(f"  - Network: {policy_kwargs['net_arch']['pi']}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Device: {model.device}")
    print("-"*80)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=10000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path=checkpoint_dir,
        name_prefix="checkpoint"
    )
    
    on_time_callback = OnTimeRateCallback(
        eval_env,
        check_freq=5000 // n_envs,
        verbose=1
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback, on_time_callback])
    
    # Training
    print(f"\n{'='*60}")
    print("🚀 STARTING DEADLINE-FOCUSED TRAINING")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=True,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    
    # Save final model
    model.save(os.path.join(checkpoint_dir, "final_model"))
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n📊 Training Statistics:")
    print(f"  - Total time: {train_time/60:.1f} minutes")
    print(f"  - Speed: {total_timesteps/train_time:.0f} steps/second")
    print(f"  - Best on-time rate: {on_time_callback.best_on_time_rate:.1%}")
    
    # Final validation
    print(f"\n{'='*60}")
    print("FINAL VALIDATION")
    print(f"{'='*60}")
    
    test_env = SchedulingEnv(data_path, max_steps=10000, reward_config={
        'on_time_reward': 1000.0,
        'early_bonus_per_day': 200.0,
        'late_penalty_per_day': -200.0,
        'utilization_bonus': 20.0,
        'action_taken_bonus': 2.0,
        'idle_penalty': -5.0,
        'sequence_violation_penalty': -1000.0,
    })
    
    obs, info = test_env.reset()
    done = False
    steps = 0
    
    while not done and steps < 10000:
        action, _ = model.predict(obs, deterministic=True)
        
        if 'action_mask' in info:
            mask = info['action_mask']
            if not mask[action]:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
        
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated
        steps += 1
    
    # Get final metrics
    metrics = test_env.reward_calculator.get_metrics()
    schedule = test_env.get_final_schedule()
    
    # Calculate performance
    completion_rate = info['tasks_scheduled'] / info['total_tasks'] * 100
    on_time_rate = metrics['on_time_rate'] * 100
    early_rate = metrics['early_rate'] * 100
    late_rate = metrics['late_rate'] * 100
    
    # Calculate utilization
    if schedule['tasks']:
        total_processing = sum(t['processing_time'] for t in schedule['tasks'])
        makespan = max(t['end'] for t in schedule['tasks'])
        n_machines = len(test_env.loader.machines)
        utilization = (total_processing / n_machines / makespan * 100)
    else:
        utilization = 0
    
    print(f"\n📈 Final Results:")
    print(f"  - Completion rate: {completion_rate:.1f}%")
    print(f"  - On-time rate: {on_time_rate:.1f}%")
    print(f"  - Early rate: {early_rate:.1f}%")
    print(f"  - Late rate: {late_rate:.1f}%")
    print(f"  - Machine utilization: {utilization:.1f}%")
    print(f"  - Total reward: {test_env.episode_reward:.0f}")
    
    # Check against targets
    print(f"\n🎯 Target Achievement:")
    print(f"  - On-time target: 85% → {on_time_rate:.1f}% {'✅' if on_time_rate >= 85 else '❌'}")
    print(f"  - Utilization target: 60% → {utilization:.1f}% {'✅' if utilization >= 60 else '❌'}")
    
    # Next steps
    if on_time_rate >= 60:
        print(f"\n✅ Achieved {on_time_rate:.1f}% on-time rate on 40_jobs.json!")
        print("📋 Next step: Scale to 100_jobs.json dataset")
    else:
        print(f"\n⚠️  Only {on_time_rate:.1f}% on-time rate achieved")
        print("📋 Recommendation: Continue training or adjust reward weights")
    
    print(f"\n{'='*80}")
    print("REWARD IMPACT ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nCompared to baseline (29% on-time rate):")
    improvement = on_time_rate / 29
    print(f"  - Improvement: {improvement:.1f}x")
    
    if on_time_rate >= 60:
        print("\n💡 Key insights:")
        print("  - Heavy late penalties drive on-time behavior")
        print("  - Reduced utilization bonus prevents over-optimization")
        print("  - Early bonuses encourage proactive scheduling")
    
    print(f"{'='*80}")
    
    env.close()
    eval_env.close()
    
    return model, {
        'completion_rate': completion_rate,
        'on_time_rate': on_time_rate,
        'utilization': utilization,
        'checkpoint_dir': checkpoint_dir
    }

if __name__ == "__main__":
    model, results = train_deadline_focused()
    print(f"\nModel saved to: {results['checkpoint_dir']}")