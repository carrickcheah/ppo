"""
Train to schedule ALL jobs with proper termination
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from phase3.environments.schedule_all_env import ScheduleAllEnvironment


def evaluate_model(model, stage_name, n_episodes=20):
    """Evaluate model performance."""
    env = ScheduleAllEnvironment(stage_name, verbose=False)
    env = Monitor(env)
    
    results = []
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            done = done or truncated
        
        scheduled = len(env.env.scheduled_jobs)
        total = env.env.total_tasks
        rate = scheduled / total
        
        results.append(rate)
        rewards.append(ep_reward)
        
        status = "✓" if rate >= 0.8 else "✗"
        print(f"Episode {ep+1}: {scheduled}/{total} = {rate:.1%} {status}, Reward: {ep_reward:.1f}")
    
    avg_rate = np.mean(results)
    return {
        'avg_rate': avg_rate,
        'avg_reward': np.mean(rewards),
        'min_rate': np.min(results),
        'max_rate': np.max(results)
    }


def train_schedule_all(stage_name='toy_normal', timesteps=200000):
    """Train to schedule all jobs."""
    
    print(f"\n{'='*60}")
    print(f"Training {stage_name} to SCHEDULE ALL JOBS")
    print(f"Episode terminates when all tasks scheduled")
    print(f"Big bonus (+200) for completing all tasks!")
    print(f"{'='*60}\n")
    
    # Create environment
    def make_env():
        env = ScheduleAllEnvironment(stage_name, verbose=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Some exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Setup checkpoints
    checkpoint_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/schedule_all/{stage_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="checkpoint"
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    print(f"\nEvaluating {stage_name}...")
    results = evaluate_model(model, stage_name)
    
    print(f"\n{'='*40}")
    print(f"RESULTS for {stage_name}:")
    print(f"  Average completion: {results['avg_rate']:.1%}")
    print(f"  Min/Max: {results['min_rate']:.1%} / {results['max_rate']:.1%}")
    print(f"  Average reward: {results['avg_reward']:.1f}")
    print(f"  Training time: {training_time/60:.1f} minutes")
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_schedule_all"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_schedule_all.zip")
    model.save(model_path)
    
    return results['avg_rate']


def main():
    """Train all stages."""
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    
    print("\nSCHEDULE ALL TRAINING")
    print("=" * 60)
    print("Key changes:")
    print("1. Episode ends when ALL tasks scheduled (not families)")
    print("2. +200 bonus for scheduling everything")
    print("3. High base reward (+30) for any scheduling")
    print("4. Small penalties for late jobs")
    print("-" * 60)
    
    results = {}
    for stage in stages:
        rate = train_schedule_all(stage, timesteps=200000)
        results[stage] = rate
        
        if rate >= 0.8:
            print(f"\n✓ {stage} ACHIEVED 80% target! ({rate:.1%})")
        else:
            print(f"\n✗ {stage}: {rate:.1%} (target: 80%)")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS - SCHEDULE ALL")
    print("="*60)
    baseline = {'toy_normal': 0.562, 'toy_hard': 0.30, 'toy_multi': 0.364}
    
    for stage in stages:
        old = baseline[stage]
        new = results[stage]
        print(f"{stage}: {old:.1%} → {new:.1%} ({new-old:+.1%})")


if __name__ == "__main__":
    main()