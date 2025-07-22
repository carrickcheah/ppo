#!/usr/bin/env python3
"""
Quick evaluation of current Phase 5 model progress
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def evaluate_current_progress():
    """Evaluate current training progress."""
    print("\n" + "="*60)
    print("Phase 5 MultiDiscrete PPO - Current Status")
    print("="*60 + "\n")
    
    # Check if any model exists
    model_paths = [
        "models/multidiscrete/best/best_model.zip",
        "models/multidiscrete/final_model.zip"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("No saved model found yet. Training may still be in early stages.")
        print("\nCreating random baseline for comparison...")
        
        # Test with random policy
        env = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=100,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=1000,
            seed=42
        )
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        invalid_actions = 0
        
        print("\nRandom Policy Baseline (100 jobs):")
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if info.get('invalid_action', False):
                invalid_actions += 1
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps taken: {steps}")
        print(f"  Jobs scheduled: {info.get('scheduled_count', 0)}/100")
        print(f"  Invalid action rate: {invalid_actions/steps*100:.1f}%")
        
        if info.get('scheduled_count', 0) >= 100:
            makespan = info.get('makespan', 0)
            print(f"  Makespan: {makespan:.1f} hours")
            
    else:
        print(f"Found model: {model_path}")
        print("Loading and evaluating...\n")
        
        # Load model
        model = PPO.load(model_path)
        
        # Create evaluation environment
        env = DummyVecEnv([lambda: MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=100,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=1000,
            seed=123
        )])
        
        # Evaluate
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        invalid_actions = 0
        
        start_time = time.time()
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            if info[0].get('invalid_action', False):
                invalid_actions += 1
        
        inference_time = time.time() - start_time
        
        print("Trained Model Results (100 jobs):")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Jobs scheduled: {info[0].get('scheduled_count', 0)}/100")
        print(f"  Invalid action rate: {invalid_actions/steps*100:.1f}%")
        print(f"  Inference time: {inference_time:.2f}s")
        
        if info[0].get('scheduled_count', 0) >= 100:
            makespan = info[0].get('makespan', 0)
            print(f"  Makespan: {makespan:.1f} hours")
    
    print("\n" + "="*60)
    print("Phase 5 Key Achievements:")
    print("- Action space reduced from 14,500 to 245 (98.3% reduction)")
    print("- Hierarchical decision making: job then machine")
    print("- MultiDiscrete compatible with SB3 PPO")
    print("- Target: <45 hours makespan (from Phase 4's 49.2h)")
    print("="*60 + "\n")

if __name__ == "__main__":
    evaluate_current_progress()