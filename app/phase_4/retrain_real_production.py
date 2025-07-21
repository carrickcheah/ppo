#!/usr/bin/env python3
"""
Retrain PPO model with real production data
Target: 130h makespan (30% above theoretical minimum of 100.3h)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from src.environments.full_production_env import FullProductionEnv
import torch.nn as nn
from datetime import datetime

def make_env(rank):
    def _init():
        env = FullProductionEnv(
            n_machines=150,
            n_jobs=500,
            state_compression="hierarchical",
            use_break_constraints=True,
            use_holiday_constraints=True,
            seed=42 + rank
        )
        return env
    return _init


def main():
    print("="*60)
    print("PHASE 4 RETRAINING WITH REAL PRODUCTION DATA")
    print("="*60)
    print("Target: 130h makespan (realistic for 14,951h workload)")
    print("="*60)

    # Configuration
    n_envs = 4  # Parallel environments
    total_timesteps = 1_000_000  # 1M steps for initial training
    save_freq = 100_000

    # Create directories
    os.makedirs("models/full_production/real_data", exist_ok=True)
    os.makedirs("logs/phase4/real_data", exist_ok=True)

    print(f"\n1. Configuration:")
    print(f"   Parallel environments: {n_envs}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Save frequency: {save_freq:,}")

    # Create training environments
    print(f"\n2. Creating {n_envs} parallel environments...")

    # Use SubprocVecEnv for true parallelism
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(9999)])

    print("\n3. Creating new PPO model...")
    # Optimized hyperparameters from extended training
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,  # Optimized
        n_steps=256,
        batch_size=1024,  # Optimized
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,  # Optimized
        policy_kwargs={
            "net_arch": [512, 512, 256],  # Deeper network for complex scheduling
            "activation_fn": nn.Tanh
        },
        verbose=1,
        device="auto",
        tensorboard_log="logs/phase4/real_data/tensorboard"
    )

    print("   Model created with optimized hyperparameters")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/full_production/real_data/checkpoints",
        name_prefix="real_data_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/full_production/real_data/best",
        log_path="logs/phase4/real_data/eval",
        eval_freq=50000,
        deterministic=True,
        render=False,
        n_eval_episodes=3
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train
    print(f"\n4. Starting training...")
    print(f"   Estimated time: 4-6 hours")
    print(f"   Press Ctrl+C to stop early\n")

    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n5. Training complete! Saving final model...")
        model.save("models/full_production/real_data/final_real_model")
        
        # Quick evaluation
        print("\n6. Quick evaluation...")
        obs = eval_env.reset()
        for _ in range(5):
            obs = eval_env.reset()
            done = False
            steps = 0
            while not done and steps < 2000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                steps += 1
                done = done[0]
                
            if 'makespan' in info[0]:
                print(f"   Makespan: {info[0]['makespan']:.1f}h")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving progress...")
        model.save("models/full_production/real_data/interrupted_model")
        print("Model saved. Run again to continue.")

    finally:
        training_time = (datetime.now() - start_time).total_seconds() / 60
        print(f"\nTraining time: {training_time:.1f} minutes")
        env.close()
        eval_env.close()

    print("\nRetraining complete!")
    print("\nNext steps:")
    print("1. Evaluate model performance")
    print("2. If <150h achieved, deploy to API")
    print("3. Otherwise, continue training with more steps")


if __name__ == "__main__":
    main()