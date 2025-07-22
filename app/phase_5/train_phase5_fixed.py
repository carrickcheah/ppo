#!/usr/bin/env python3
"""
Fixed Phase 5 training with proper job count handling
No max_valid_actions limitation - uses all 411 jobs
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def train_phase5_fixed():
    print("\n" + "="*60)
    print("Phase 5 Fixed Training - All Jobs Visible")
    print("="*60 + "\n")
    
    # Create environment WITHOUT max_valid_actions limitation
    print("Creating environment with ALL jobs accessible...")
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,  # Use all jobs from snapshot
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=1000,
        use_break_constraints=True,  # Enable for realistic training
        use_holiday_constraints=True,
        invalid_action_penalty=-10.0,
        seed=42
    )
    
    # Reset to check actual dimensions
    obs, info = env.reset()
    print(f"\nEnvironment created successfully:")
    print(f"  Action space: {env.action_space}")
    print(f"  Action dimensions: {info.get('action_space_dims', 'unknown')}")
    print(f"  Jobs available: {len(env.jobs) if hasattr(env, 'jobs') else 'unknown'}")
    print(f"  Machines available: {len(env.machines) if hasattr(env, 'machines') else 'unknown'}")
    print(f"  Internal n_jobs: {env.n_jobs if hasattr(env, 'n_jobs') else 'unknown'}")
    
    # Wrap for training
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    # Create model
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="models/multidiscrete/fixed",
        name_prefix="phase5_fixed"
    )
    
    # Train
    print("\nStarting training for 200,000 steps...")
    print("Expected time: 5-10 minutes\n")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=200000,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Save final model
    os.makedirs("models/multidiscrete/fixed", exist_ok=True)
    model.save("models/multidiscrete/fixed/final_model")
    print("Model saved to: models/multidiscrete/fixed/final_model.zip")
    
    # Test the trained model
    print("\n" + "="*40)
    print("Testing trained model...")
    print("="*40)
    
    # Fresh environment for testing
    test_env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=1000,
        seed=123
    )
    test_vec_env = DummyVecEnv([lambda: test_env])
    
    obs = test_vec_env.reset()
    total_reward = 0
    steps = 0
    scheduled = 0
    invalid = 0
    
    print("\nRunning test episode...")
    while steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_vec_env.step(action)
        total_reward += reward[0]
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid += 1
        
        scheduled = info[0].get('scheduled_count', 0)
        
        # Progress updates
        if steps % 100 == 0:
            invalid_rate = invalid/steps*100
            print(f"  Step {steps}: {scheduled} jobs scheduled, {invalid_rate:.1f}% invalid")
            
        if done[0]:
            print(f"\nEpisode completed!")
            break
    
    print(f"\n" + "="*40)
    print(f"Final Results:")
    print(f"="*40)
    print(f"  Steps taken: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Jobs scheduled: {scheduled}/411")
    print(f"  Invalid action rate: {invalid/steps*100:.1f}%")
    
    if scheduled == 411:
        makespan = info[0].get('makespan', 0)
        print(f"  Makespan: {makespan:.1f} hours")
        print(f"\n✅ SUCCESS! All 411 jobs scheduled!")
        print(f"Compare to Phase 4: 49.2 hours (target: <45 hours)")
        
        if makespan < 45:
            print(f"\n🎉 TARGET ACHIEVED! Makespan {makespan:.1f}h < 45h")
            improvement = (49.2 - makespan) / 49.2 * 100
            print(f"Improvement: {improvement:.1f}% reduction from Phase 4")
    else:
        completion_rate = scheduled/411*100
        print(f"\n⚠️  Partial completion: {completion_rate:.1f}%")
        print(f"Need more training to schedule all jobs")
    
    print("\n" + "="*60)
    print("Phase 5 Fixed Training Complete!")
    print("Next: Run full 2M timestep training for best results")
    print("="*60)

if __name__ == "__main__":
    train_phase5_fixed()