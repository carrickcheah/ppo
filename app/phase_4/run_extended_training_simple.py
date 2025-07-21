#!/usr/bin/env python3
"""
Simple Extended Training Script for Phase 4

This script runs extended training using the existing Phase 4 configuration.
"""

import os
import sys
import time
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from environments.full_production_env import FullProductionEnv


def make_env(env_id: int = 0):
    """Create a full production environment."""
    def _init():
        env = FullProductionEnv(
            n_machines=152,
            n_jobs=500,
            data_file="data/parsed_production_data_boolean.json",
            use_break_constraints=True,
            use_holiday_constraints=True,
            state_compression="hierarchical",
            seed=42 + env_id
        )
        return env
    return _init


def main():
    print("="*60)
    print("PHASE 4 EXTENDED TRAINING (2M TIMESTEPS)")
    print("="*60)
    print("Current performance: 49.2h makespan")
    print("Target: <45h makespan")
    print("Training approach: Extended training with optimized hyperparameters")
    print("-"*60)
    
    # Create directories
    os.makedirs("models/full_production/extended", exist_ok=True)
    os.makedirs("models/full_production/extended/checkpoints", exist_ok=True)
    os.makedirs("logs/phase4/extended", exist_ok=True)
    
    # Create environments
    print("\nCreating environments...")
    n_envs = 4  # Use 4 parallel environments
    env_fns = [make_env(i) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)
    eval_env = DummyVecEnv([make_env(99)])
    
    # Load existing model
    print("\nLoading Phase 4 model from 1M timesteps...")
    model_path = "models/full_production/final_model.zip"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    model = PPO.load(model_path, env=train_env)
    
    # Update hyperparameters
    print("\nUpdating hyperparameters for extended training:")
    print("- Learning rate: 1e-5 → 3e-5 (3x)")
    print("- Batch size: 512 → 1024 (2x)")
    print("- Entropy coefficient: 0.01 → 0.005 (half)")
    
    model.learning_rate = 3e-5
    model.batch_size = 1024
    model.ent_coef = 0.005
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=250000,  # Every 250k steps
        save_path="models/full_production/extended/checkpoints",
        name_prefix="extended_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/full_production/extended/best_model",
        log_path="logs/phase4/extended",
        eval_freq=50000,
        n_eval_episodes=3,
        deterministic=True
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print("\nStarting extended training...")
    print("This will take approximately 20 hours")
    print("Checkpoints will be saved every 250k steps")
    print("\nPress Ctrl+C to stop training and save progress")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=1000000,  # Additional 1M steps (total 2M)
            callback=callbacks,
            log_interval=100,
            tb_log_name="phase4_extended",
            reset_num_timesteps=False,  # Continue from existing
            progress_bar=True
        )
        
        # Save final model
        print("\nSaving final extended model...")
        model.save("models/full_production/extended/final_extended_model")
        
        elapsed = (time.time() - start_time) / 3600
        print(f"\nTraining completed in {elapsed:.1f} hours")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current model...")
        model.save("models/full_production/extended/interrupted_model")
        print("Model saved. You can resume training later.")
    
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()