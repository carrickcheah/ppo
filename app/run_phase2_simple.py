"""
Simple Phase 2 training script with better progress tracking.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.environments.scaled_production_env import ScaledProductionEnv


def main():
    print("\n" + "="*60)
    print("PHASE 2 TRAINING: ADDING BREAK CONSTRAINTS")
    print("="*60)
    
    # Check Phase 1 model
    phase1_path = "models/curriculum/phase1_no_breaks/final_model"
    if not Path(phase1_path + ".zip").exists():
        print("ERROR: Phase 1 model not found!")
        return
    
    # Create Phase 2 environment with breaks
    print("Creating environment with break constraints...")
    env = Monitor(ScaledProductionEnv(
        n_machines=40,
        use_break_constraints=True,  # Enable breaks
        seed=42
    ))
    
    # Load Phase 1 model with new environment
    print("\nLoading Phase 1 model...")
    model = PPO.load(phase1_path, env=env)
    model.learning_rate = 5e-5  # Reduce for fine-tuning
    
    print("\nTraining configuration:")
    print(f"- Learning rate: {model.learning_rate}")
    print("- Break constraints: ENABLED")
    print("- Training steps: 500,000 (shorter for testing)")
    
    # Quick evaluation before training
    print("\nEvaluating Phase 1 model on Phase 2 environment...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    initial_makespan = env.unwrapped.episode_makespan
    print(f"Initial performance with breaks: {initial_makespan:.1f}h makespan")
    print(f"(Phase 1 without breaks was: 16.2h)")
    
    # Train with progress tracking
    print("\nStarting Phase 2 training...")
    print("-" * 60)
    
    start_time = datetime.now()
    total_timesteps = 500000  # Shorter for testing
    
    # Custom training loop for better progress tracking
    n_steps = 0
    episode_rewards = []
    episode_makespans = []
    
    while n_steps < total_timesteps:
        # Collect experience
        model.learn(
            total_timesteps=10000,  # Small batches
            reset_num_timesteps=False,
            progress_bar=False
        )
        n_steps += 10000
        
        # Evaluate current performance
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        makespan = env.unwrapped.episode_makespan
        episode_rewards.append(total_reward)
        episode_makespans.append(makespan)
        
        # Progress update every 50k steps
        if n_steps % 50000 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            avg_reward = np.mean(episode_rewards[-5:]) if episode_rewards else 0
            avg_makespan = np.mean(episode_makespans[-5:]) if episode_makespans else 0
            
            print(f"\nProgress: {n_steps:,}/{total_timesteps:,} steps ({n_steps/total_timesteps*100:.1f}%)")
            print(f"Time elapsed: {elapsed:.1f} minutes")
            print(f"Recent avg reward: {avg_reward:.1f}")
            print(f"Recent avg makespan: {avg_makespan:.1f}h")
            
            # Check improvement
            if episode_makespans:
                improvement = (initial_makespan - avg_makespan) / initial_makespan * 100
                print(f"Improvement from initial: {improvement:.1f}%")
    
    # Final evaluation
    print("\n" + "="*60)
    print("PHASE 2 TRAINING COMPLETE")
    print("="*60)
    
    # Evaluate final performance
    final_makespans = []
    final_rewards = []
    
    for i in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        final_makespans.append(env.unwrapped.episode_makespan)
        final_rewards.append(total_reward)
        print(f"Eval {i+1}: Makespan={env.unwrapped.episode_makespan:.1f}h, Reward={total_reward:.1f}")
    
    # Save model
    phase2_path = "models/curriculum/phase2_with_breaks/final_model"
    Path(phase2_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(phase2_path)
    
    # Summary
    print("\n" + "-"*60)
    print("RESULTS SUMMARY:")
    print(f"Phase 1 (no breaks): 16.2h")
    print(f"Phase 2 initial (with breaks): {initial_makespan:.1f}h")
    print(f"Phase 2 final (with breaks): {np.mean(final_makespans):.1f}h")
    print(f"Break constraint penalty: {(np.mean(final_makespans) - 16.2) / 16.2 * 100:.1f}%")
    print(f"\nModel saved to: {phase2_path}")
    
    # Compare with baselines
    print("\nBaseline comparisons:")
    print("- Random policy baseline: ~19.4h")
    print("- Previous PPO with breaks: 21.9h")
    print(f"- Phase 2 curriculum: {np.mean(final_makespans):.1f}h")
    
    if np.mean(final_makespans) < 19.4:
        print("\n✓ SUCCESS: Phase 2 beats random baseline even with breaks!")
    else:
        print("\n⚠ Phase 2 needs more training to beat baseline with breaks")


if __name__ == "__main__":
    main()