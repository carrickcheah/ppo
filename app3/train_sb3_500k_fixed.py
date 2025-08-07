#!/usr/bin/env python
"""
Train SB3 PPO for 500k steps on single dataset
Focus on one dataset to avoid observation space mismatch
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environments.scheduling_env import SchedulingEnv
import torch
import numpy as np
import time
from datetime import datetime

def make_env(data_path: str, rank: int = 0):
    """Create environment for SB3."""
    def _init():
        env = SchedulingEnv(
            snapshot_path=data_path,
            max_steps=5000,
            planning_horizon=720.0,
            reward_config={
                'on_time_reward': 500.0,      # 5x bonus
                'early_bonus_per_day': 200.0,  # 4x bonus
                'late_penalty_per_day': -20.0, # Reduced penalty
                'utilization_bonus': 150.0,    # 15x bonus
                'action_taken_bonus': 50.0,    # 10x bonus
                'idle_penalty': -10.0,          # 10x penalty
                'sequence_violation_penalty': -50.0,
            }
        )
        env.reset(seed=42 + rank)
        return Monitor(env)
    return _init

def train_sb3_500k_single():
    """Train SB3 PPO for 500k steps on 100 jobs dataset."""
    
    print("="*80)
    print("SB3 PPO INTENSIVE TRAINING - 500K STEPS")
    print("="*80)
    print("\nTraining on 100 jobs dataset for maximum performance")
    print("Target: 10x improvement over custom PPO")
    print("-"*80)
    
    # Create directories
    os.makedirs("checkpoints/sb3_500k", exist_ok=True)
    os.makedirs("logs/sb3_500k", exist_ok=True)
    
    data_path = 'data/100_jobs.json'
    total_timesteps = 500000
    
    print(f"\nDataset: {data_path}")
    print(f"Total steps: {total_timesteps:,}")
    print(f"Estimated time: ~15 minutes")
    print("-"*80)
    
    # Create parallel environments for faster training
    n_envs = 4  # Use 4 parallel environments
    print(f"\nUsing {n_envs} parallel environments for faster training")
    
    env = make_vec_env(
        make_env(data_path),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )
    
    # Create eval environment
    eval_env = make_vec_env(
        make_env(data_path, rank=100),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    print("\nCreating ENHANCED SB3 PPO model...")
    print("-"*40)
    
    # Enhanced network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[2048, 1024, 512, 256],  # Very large policy network
            vf=[2048, 1024, 512, 256]   # Separate value network
        ),
        activation_fn=torch.nn.GELU,    # Better activation
        ortho_init=True,
        share_features_extractor=False  # Separate networks
    )
    
    # Optimized hyperparameters for scheduling
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lambda f: 3e-4 * (1 - f),  # Linear decay
        n_steps=2048,            # Steps per environment
        batch_size=256,          # Larger batches
        n_epochs=20,             # More training epochs
        gamma=0.998,             # Very high gamma for long-term
        gae_lambda=0.99,         # High GAE
        clip_range=lambda f: 0.2 * (1 - 0.5 * f),  # Decay clipping
        clip_range_vf=None,
        ent_coef=0.05,           # High exploration initially
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,          # Higher KL for more updates
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/sb3_500k",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("Model Configuration:")
    print(f"  - Architecture: {policy_kwargs['net_arch']}")
    print(f"  - Activation: GELU")
    print(f"  - Parameters: ~8M")
    print(f"  - Learning rate: 3e-4 with linear decay")
    print(f"  - Rollout: {n_envs} × 2048 = {n_envs * 2048} steps")
    print(f"  - Device: {model.device}")
    print("-"*80)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/sb3_500k",
        log_path="logs/sb3_500k",
        eval_freq=25000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path="checkpoints/sb3_500k",
        name_prefix="checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Training
    print(f"\n{'='*60}")
    print("STARTING INTENSIVE TRAINING")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=True,
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current progress...")
    
    train_time = time.time() - start_time
    
    # Save final model
    model.save("checkpoints/sb3_500k/final_500k_model")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n📊 Training Statistics:")
    print(f"  - Total time: {train_time:.1f}s ({train_time/60:.1f} minutes)")
    print(f"  - Training speed: {total_timesteps/train_time:.0f} steps/second")
    print(f"  - Model saved: checkpoints/sb3_500k/final_500k_model.zip")
    
    # Validation
    print(f"\n{'='*60}")
    print("VALIDATION ON 100 JOBS")
    print(f"{'='*60}")
    
    test_env = SchedulingEnv(data_path, max_steps=10000)
    obs, info = test_env.reset()
    
    val_start = time.time()
    done = False
    steps = 0
    
    while not done and steps < 10000:
        action, _ = model.predict(obs, deterministic=True)
        
        # Handle action masking
        if 'action_mask' in info:
            mask = info['action_mask']
            if not mask[action]:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
        
        obs, reward, terminated, truncated, info = test_env.step(int(action))
        done = terminated or truncated
        steps += 1
    
    val_time = time.time() - val_start
    
    # Calculate metrics
    schedule = test_env.get_final_schedule()
    if schedule['tasks']:
        total_processing = sum(t['processing_time'] for t in schedule['tasks'])
        makespan = max(t['end'] for t in schedule['tasks'])
        n_machines = len(test_env.loader.machines)
        efficiency = (total_processing / n_machines / makespan * 100)
        
        late_jobs = sum(1 for t in schedule['tasks'] if t['end'] > t['lcd_days'] * 24)
        on_time_rate = (1 - late_jobs / len(schedule['tasks'])) * 100
    else:
        efficiency = 0
        on_time_rate = 0
    
    completion_rate = info['tasks_scheduled'] / info['total_tasks'] * 100
    
    print(f"\n📈 Performance Metrics:")
    print(f"  - Completion: {info['tasks_scheduled']}/{info['total_tasks']} ({completion_rate:.1f}%)")
    print(f"  - Efficiency: {efficiency:.1f}%")
    print(f"  - On-time delivery: {on_time_rate:.1f}%")
    print(f"  - Episode reward: {test_env.episode_reward:.0f}")
    print(f"  - Makespan: {makespan:.1f} hours")
    print(f"  - Steps: {steps}")
    print(f"  - Time: {val_time:.2f}s")
    
    # Compare with baselines
    print(f"\n{'='*60}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*60}")
    
    custom_efficiency = 7.4  # From validation
    demo_efficiency = 3.1    # 50k model
    
    improvement_vs_custom = efficiency / custom_efficiency
    improvement_vs_demo = efficiency / demo_efficiency
    
    print(f"\n📊 Efficiency Comparison:")
    print(f"  - Custom PPO (baseline): {custom_efficiency:.1f}%")
    print(f"  - SB3 50k (demo): {demo_efficiency:.1f}%")
    print(f"  - SB3 500k (current): {efficiency:.1f}%")
    
    print(f"\n🎯 Improvement Factors:")
    print(f"  - vs Custom PPO: {improvement_vs_custom:.1f}x")
    print(f"  - vs 50k Demo: {improvement_vs_demo:.1f}x")
    
    if improvement_vs_custom >= 10:
        print("\n🏆 SUCCESS! 10x IMPROVEMENT ACHIEVED!")
        print("   Ready for production deployment!")
    elif improvement_vs_custom >= 5:
        print("\n✅ EXCELLENT! 5x+ improvement achieved!")
        print("   Continue training for 100x goal")
    else:
        print(f"\n📈 Good progress: {improvement_vs_custom:.1f}x improvement")
        print("   More training needed for target")
    
    print(f"\n{'='*80}")
    print("PATH TO 100x:")
    print(f"{'='*80}")
    print(f"Current: {improvement_vs_custom:.1f}x")
    print(f"Next milestones:")
    print(f"  - 1M steps: Expected ~15-20x")
    print(f"  - 2M steps: Expected ~30-40x")
    print(f"  - 5M steps: Expected ~60-80x")
    print(f"  - 10M steps: Target 100x")
    print(f"{'='*80}")
    
    env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    model = train_sb3_500k_single()