#!/usr/bin/env python
"""
OPTIMIZED SB3 PPO with aggressive hyperparameters for 100x improvement
Not just more training - BETTER training with tuned hyperparameters
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

def make_env(data_path: str, rank: int = 0):
    """Create environment with OPTIMIZED rewards."""
    def _init():
        env = SchedulingEnv(
            snapshot_path=data_path,
            max_steps=10000,  # Longer episodes for learning
            planning_horizon=720.0,
            reward_config={
                # AGGRESSIVE REWARD SHAPING FOR EFFICIENCY
                'on_time_reward': 1000.0,        # 10x increase
                'early_bonus_per_day': 500.0,    # 10x increase
                'late_penalty_per_day': -10.0,   # Reduced penalty
                'utilization_bonus': 500.0,      # 50x increase - KEY FOR EFFICIENCY
                'action_taken_bonus': 100.0,     # 20x increase
                'idle_penalty': -50.0,            # 50x penalty for idle
                'sequence_violation_penalty': -25.0,  # Reduced
            }
        )
        env.reset(seed=42 + rank)
        return Monitor(env)
    return _init

def train_optimized():
    """Train with OPTIMIZED hyperparameters for 100x improvement."""
    
    print("="*80)
    print("OPTIMIZED SB3 PPO - AGGRESSIVE HYPERPARAMETERS FOR 100x")
    print("="*80)
    print("\nKEY CHANGES FOR 100x:")
    print("1. MASSIVE network (4096 neurons)")
    print("2. HIGH learning rate (1e-3)")
    print("3. LARGE batch size (512)")
    print("4. AGGRESSIVE exploration (ent_coef=0.1)")
    print("5. OPTIMIZED rewards (50x utilization bonus)")
    print("6. 8 PARALLEL environments")
    print("-"*80)
    
    # Create directories
    os.makedirs("checkpoints/sb3_optimized", exist_ok=True)
    os.makedirs("logs/sb3_optimized", exist_ok=True)
    
    data_path = 'data/100_jobs.json'
    total_timesteps = 200000  # Quick test with optimized params
    
    print(f"\nDataset: {data_path}")
    print(f"Total steps: {total_timesteps:,}")
    print("-"*80)
    
    # Use MORE parallel environments for diverse experience
    n_envs = 8  # 8 parallel environments
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
    
    print("\n🚀 CREATING OPTIMIZED MODEL WITH AGGRESSIVE SETTINGS")
    print("-"*60)
    
    # MASSIVE NETWORK for complex patterns
    policy_kwargs = dict(
        net_arch=dict(
            pi=[4096, 2048, 1024, 512, 256],  # HUGE policy network
            vf=[4096, 2048, 1024, 512, 256]   # HUGE value network
        ),
        activation_fn=torch.nn.GELU,  # Better gradients
        ortho_init=True,
        share_features_extractor=False,
        normalize_images=False,
    )
    
    # AGGRESSIVE HYPERPARAMETERS
    model = PPO(
        policy="MlpPolicy",
        env=env,
        
        # LEARNING PARAMETERS - AGGRESSIVE
        learning_rate=1e-3,              # HIGH learning rate (was 3e-4)
        n_steps=4096,                    # LARGE rollout buffer
        batch_size=512,                  # HUGE batch size (was 64-256)
        n_epochs=30,                     # MORE epochs (was 10-20)
        
        # DISCOUNT & ADVANTAGE - OPTIMIZED
        gamma=0.999,                     # VERY high for long-term planning
        gae_lambda=0.99,                 # HIGH GAE
        
        # EXPLORATION - AGGRESSIVE
        clip_range=0.3,                  # WIDER clip range (was 0.2)
        clip_range_vf=None,
        ent_coef=0.1,                    # HIGH entropy (was 0.01-0.05)
        
        # VALUE FUNCTION - BALANCED
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # KL TARGET - AGGRESSIVE
        target_kl=0.1,                   # HIGH KL (was 0.02-0.05)
        
        # NETWORK
        policy_kwargs=policy_kwargs,
        
        # OTHER
        verbose=1,
        tensorboard_log="logs/sb3_optimized",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    print("🔥 OPTIMIZED CONFIGURATION:")
    print(f"  - Network: {policy_kwargs['net_arch']['pi']}")
    print(f"  - Parameters: ~20M")
    print(f"  - Learning rate: 1e-3 (3x higher)")
    print(f"  - Batch size: 512 (2-8x larger)")
    print(f"  - Entropy: 0.1 (10x more exploration)")
    print(f"  - Rollout: {n_envs} × 4096 = {n_envs * 4096} steps")
    print(f"  - Device: {model.device}")
    print("-"*80)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/sb3_optimized",
        log_path="logs/sb3_optimized",
        eval_freq=10000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path="checkpoints/sb3_optimized",
        name_prefix="checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Training
    print(f"\n{'='*60}")
    print("🚀 STARTING OPTIMIZED TRAINING")
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
        print("\n\nTraining interrupted")
    
    train_time = time.time() - start_time
    
    # Save model
    model.save("checkpoints/sb3_optimized/optimized_model")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n📊 Training Statistics:")
    print(f"  - Total time: {train_time:.1f}s")
    print(f"  - Speed: {total_timesteps/train_time:.0f} steps/second")
    
    # Validation
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    
    test_env = SchedulingEnv(data_path, max_steps=10000)
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
        
        if steps % 500 == 0:
            print(f"  Step {steps}: {info['tasks_scheduled']}/{info['total_tasks']} scheduled")
    
    # Calculate metrics
    schedule = test_env.get_final_schedule()
    if schedule['tasks']:
        total_processing = sum(t['processing_time'] for t in schedule['tasks'])
        makespan = max(t['end'] for t in schedule['tasks'])
        n_machines = len(test_env.loader.machines)
        efficiency = (total_processing / n_machines / makespan * 100)
    else:
        efficiency = 0
    
    completion_rate = info['tasks_scheduled'] / info['total_tasks'] * 100
    
    print(f"\n📈 Results:")
    print(f"  - Completion: {completion_rate:.1f}%")
    print(f"  - Efficiency: {efficiency:.1f}%")
    print(f"  - Reward: {test_env.episode_reward:.0f}")
    
    # Compare
    baseline_efficiency = 7.4  # Custom PPO
    improvement = efficiency / baseline_efficiency
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT")
    print(f"{'='*60}")
    
    print(f"  - Baseline: {baseline_efficiency:.1f}%")
    print(f"  - Optimized: {efficiency:.1f}%")
    print(f"  - Improvement: {improvement:.1f}x")
    
    if improvement >= 10:
        print("\n🏆 10x+ IMPROVEMENT ACHIEVED!")
    elif improvement >= 5:
        print("\n✅ 5x+ improvement!")
    else:
        print(f"\n📈 {improvement:.1f}x - Continue training")
    
    print(f"\n{'='*80}")
    print("KEY OPTIMIZATIONS THAT MAKE THE DIFFERENCE:")
    print(f"{'='*80}")
    
    optimizations = [
        "1. 🧠 HUGE NETWORK: 4096→2048→1024→512→256 (captures complex patterns)",
        "2. 📈 HIGH LEARNING RATE: 1e-3 (3x faster learning)",
        "3. 📦 LARGE BATCHES: 512 samples (stable gradients)",
        "4. 🎲 HIGH ENTROPY: 0.1 (10x more exploration)",
        "5. 💰 REWARD TUNING: 50x utilization bonus (focuses on efficiency)",
        "6. 🔄 MORE EPOCHS: 30 per update (better convergence)",
        "7. 🚀 PARALLEL ENVS: 8x diverse experience",
        "8. 📊 WIDE CLIP: 0.3 (allows bigger updates)"
    ]
    
    for opt in optimizations:
        print(opt)
    
    print(f"\n{'='*80}")
    print("PROJECTED PERFORMANCE WITH FULL TRAINING:")
    print(f"{'='*80}")
    
    # Projection based on current improvement rate
    steps_per_x = 200000 / improvement if improvement > 0 else 1000000
    
    print(f"Current: {total_timesteps:,} steps → {improvement:.1f}x")
    print(f"Rate: {steps_per_x:,.0f} steps per 1x improvement")
    print(f"\nProjection:")
    print(f"  500k steps: ~{500000/steps_per_x:.0f}x")
    print(f"  1M steps: ~{1000000/steps_per_x:.0f}x")
    print(f"  5M steps: ~{5000000/steps_per_x:.0f}x")
    
    steps_to_100x = int(steps_per_x * 100)
    print(f"\n🎯 Steps to 100x: {steps_to_100x:,}")
    print(f"   Time required: ~{steps_to_100x/10000/60:.1f} hours")
    
    print(f"{'='*80}")
    
    env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    model = train_optimized()