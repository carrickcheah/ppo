#!/usr/bin/env python3
"""
Diagnose why the model isn't learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def diagnose_training():
    print("\n" + "="*60)
    print("Diagnosing Training Issues")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=100,
        seed=42
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load the trained model
    model = PPO.load("models/multidiscrete/working/demo_model")
    
    print("1. Model Architecture:")
    print(f"   Policy network: {model.policy}")
    
    # Get a batch of observations
    obs = vec_env.reset()
    
    print("\n2. Model Predictions Analysis:")
    
    # Get 10 predictions
    print("   First 10 action predictions:")
    for i in range(10):
        with torch.no_grad():
            # Get raw logits from the model
            obs_tensor = torch.as_tensor(obs).float()
            features = model.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            action_logits = model.policy.action_net(latent_pi)
            
            # The action space is MultiDiscrete([411, 145])
            # So we should have 411 + 145 = 556 logits
            print(f"   Step {i}: Logits shape: {action_logits.shape}")
            
            # Split logits for job and machine selection
            job_logits = action_logits[:, :411]
            machine_logits = action_logits[:, 411:]
            
            # Get probabilities
            job_probs = torch.softmax(job_logits, dim=-1)
            machine_probs = torch.softmax(machine_logits, dim=-1)
            
            # Get most likely actions
            job_action = torch.argmax(job_probs, dim=-1).item()
            machine_action = torch.argmax(machine_probs, dim=-1).item()
            
            print(f"      Predicted: job={job_action}, machine={machine_action}")
            print(f"      Job prob: {job_probs[0, job_action]:.3f}, Machine prob: {machine_probs[0, machine_action]:.3f}")
        
        # Step environment with predicted action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        invalid = info[0].get('invalid_action', False)
        reason = info[0].get('invalid_reason', 'N/A')
        print(f"      Result: reward={reward[0]:.1f}, invalid={invalid}")
        if invalid:
            print(f"      Reason: {reason}")
        print()
        
        if done[0]:
            obs = vec_env.reset()
    
    # Check reward statistics during training
    print("\n3. Training Diagnostics:")
    print("   Check if rewards are too sparse")
    print("   Invalid action penalty: -5.0")
    print("   Valid action rewards should be positive")
    
    # Test with exploration
    print("\n4. Testing with Exploration (stochastic policy):")
    obs = vec_env.reset()
    valid_found = 0
    
    for i in range(20):
        action, _ = model.predict(obs, deterministic=False)  # Stochastic
        obs, reward, done, info = vec_env.step(action)
        
        if not info[0].get('invalid_action', False):
            valid_found += 1
            print(f"   Step {i}: Found valid action! Reward: {reward[0]:.2f}")
            scheduled = info[0].get('scheduled_count', 0)
            print(f"   Jobs scheduled: {scheduled}")
        
        if done[0]:
            obs = vec_env.reset()
    
    print(f"\n   Valid actions with exploration: {valid_found}/20")
    
    # Check compatibility matrix
    print("\n5. Environment Compatibility Check:")
    env.reset()
    if hasattr(env, 'compatibility_matrix') and env.compatibility_matrix is not None:
        total_compatible = np.sum(env.compatibility_matrix)
        avg_per_job = np.mean(np.sum(env.compatibility_matrix, axis=1))
        print(f"   Total compatible pairs: {total_compatible}")
        print(f"   Average machines per job: {avg_per_job:.1f}")
        
        # Check if first job has any compatible machines
        first_job_compatible = np.sum(env.compatibility_matrix[0])
        print(f"   First job compatible machines: {first_job_compatible}")
        if first_job_compatible > 0:
            compatible_machines = np.where(env.compatibility_matrix[0])[0]
            print(f"   Compatible machine indices: {compatible_machines[:5]}...")

if __name__ == "__main__":
    diagnose_training()