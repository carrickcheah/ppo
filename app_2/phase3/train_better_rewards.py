"""
Train with better reward structure that encourages scheduling all jobs
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

from phase3.environments.better_reward_env import BetterRewardEnvironment


def evaluate_model(model, stage_name, n_episodes=20):
    """Evaluate model performance."""
    env = BetterRewardEnvironment(stage_name, verbose=False)
    env = Monitor(env)
    
    results = {
        'scheduled': [],
        'rewards': [],
        'late_jobs': []
    }
    
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
        
        results['scheduled'].append(scheduled)
        results['rewards'].append(ep_reward)
        
        print(f"Episode {ep+1}: {scheduled}/{total} = {scheduled/total:.1%}, Reward: {ep_reward:.1f}")
    
    avg_rate = np.mean([s/env.env.total_tasks for s in results['scheduled']])
    return {
        'avg_rate': avg_rate,
        'avg_reward': np.mean(results['rewards']),
        'min_rate': min(s/env.env.total_tasks for s in results['scheduled']),
        'max_rate': max(s/env.env.total_tasks for s in results['scheduled'])
    }


def train_with_better_rewards(stage_name='toy_normal', timesteps=300000):
    """Train using better reward structure."""
    
    print(f"\n{'='*60}")
    print(f"Training {stage_name} with BETTER REWARDS")
    print(f"Goal: Schedule ALL jobs, even if some are late")
    print(f"{'='*60}\n")
    
    # Create environment
    def make_env():
        env = BetterRewardEnvironment(stage_name, verbose=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create model with same hyperparameters as successful training
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
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Setup checkpoints
    checkpoint_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/better_rewards/{stage_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="checkpoint"
    )
    
    # Train
    print("Starting training with better rewards...")
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
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_better_rewards"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_better_rewards.zip")
    model.save(model_path)
    
    # Save results
    results_data = {
        'stage': stage_name,
        'avg_rate': results['avg_rate'],
        'min_rate': results['min_rate'],
        'max_rate': results['max_rate'],
        'avg_reward': results['avg_reward'],
        'training_time_min': training_time/60,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, f"{stage_name}_results.json"), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return results['avg_rate']


def main():
    """Train all stages with better rewards."""
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    
    print("\nBETTER REWARD TRAINING")
    print("=" * 60)
    print("Key changes:")
    print("1. Always positive reward for scheduling (+20 base)")
    print("2. Proportional late penalty (not binary -1000)")
    print("3. Big bonus (+100) for scheduling ALL jobs")
    print("4. Progress bonuses at 50% and 75% completion")
    print("-" * 60)
    
    results = {}
    for stage in stages:
        rate = train_with_better_rewards(stage, timesteps=300000)
        results[stage] = rate
        
        if rate >= 0.8:
            print(f"\n✓ {stage} ACHIEVED 80% target! ({rate:.1%})")
        else:
            print(f"\n✗ {stage}: {rate:.1%} (target: 80%)")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS - BETTER REWARDS")
    print("="*60)
    baseline = {'toy_normal': 0.562, 'toy_hard': 0.30, 'toy_multi': 0.364}
    
    for stage in stages:
        old = baseline[stage]
        new = results[stage]
        print(f"{stage}: {old:.1%} → {new:.1%} ({new-old:+.1%})")


if __name__ == "__main__":
    main()