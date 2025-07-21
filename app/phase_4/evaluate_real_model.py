#!/usr/bin/env python3
"""
Comprehensive evaluation of the retrained model on real production data
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

print("="*60)
print("EVALUATING RETRAINED MODEL ON REAL DATA")
print("="*60)

# Model paths to evaluate
models = [
    ("Final Real Model (1M steps)", "models/full_production/real_data/final_real_model.zip"),
    ("Best Model (from eval)", "models/full_production/real_data/best/best_model.zip"),
    ("Original Synthetic Model", "models/full_production/final_model.zip")
]

# Create environment
print("\n1. Creating evaluation environment...")
env = FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)

print(f"   Environment loaded:")
print(f"   - Machines: {len(env.machines)}")
print(f"   - Jobs: {len(env.jobs)}")
print(f"   - Total workload: {sum(j['processing_time'] for j in env.jobs):.1f}h")
print(f"   - Theoretical minimum: {sum(j['processing_time'] for j in env.jobs)/len(env.machines):.1f}h")

# Evaluate each model
print("\n2. Evaluating models...")

for model_name, model_path in models:
    if not os.path.exists(model_path):
        print(f"\n   {model_name}: Not found")
        continue
        
    print(f"\n   {model_name}:")
    model = PPO.load(model_path)
    
    # Run 3 evaluation episodes
    results = []
    for episode in range(3):
        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        episode_reward = 0
        actions_taken = []
        
        while not (terminated or truncated) and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        # Analyze results
        scheduled_jobs = 0
        total_jobs = len(env.jobs)
        
        # Count how many jobs were actually scheduled
        if hasattr(env, 'scheduled_jobs'):
            scheduled_jobs = len(env.scheduled_jobs)
        elif hasattr(env, 'completed_tasks'):
            scheduled_jobs = sum(len(tasks) for tasks in env.completed_tasks.values())
        
        makespan = info.get('makespan', 0)
        completion_rate = scheduled_jobs / total_jobs if total_jobs > 0 else 0
        
        results.append({
            'makespan': makespan,
            'scheduled': scheduled_jobs,
            'completion': completion_rate,
            'steps': steps,
            'reward': episode_reward,
            'unique_actions': len(np.unique(actions_taken))
        })
        
        print(f"     Episode {episode+1}: {makespan:.1f}h, {scheduled_jobs}/{total_jobs} jobs, {steps} steps")
    
    # Summary statistics
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_scheduled = np.mean([r['scheduled'] for r in results])
    avg_completion = np.mean([r['completion'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"     Average makespan: {avg_makespan:.1f}h")
    print(f"     Average jobs scheduled: {avg_scheduled:.0f}/{total_jobs}")
    print(f"     Average completion rate: {avg_completion:.1%}")
    print(f"     Average episode length: {avg_steps:.0f} steps")

env.close()

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

print("\nKey observations:")
print("1. Episode length of 172 steps is very short for 411 jobs")
print("2. This suggests the model is terminating early")
print("3. The 15.9h makespan likely represents partial scheduling")

print("\nPossible issues:")
print("- Environment may be truncating episodes too early")
print("- Model learned to terminate rather than continue scheduling")
print("- Reward structure may encourage early termination")

print("\nRecommendations:")
print("1. Check max_episode_steps in environment (currently 2000)")
print("2. Verify all jobs are being presented as valid actions")
print("3. Consider training with higher max_episode_steps")
print("4. Check if reward penalizes incomplete scheduling")