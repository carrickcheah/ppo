#!/usr/bin/env python3
"""
Test current model on scaled production data
Scale down real job durations to match training distribution
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

print("="*60)
print("TESTING MODEL ON SCALED PRODUCTION DATA")
print("="*60)

# First, create a scaled version of the data
print("\n1. Creating scaled production data...")

# Load real data
with open('data/real_production_snapshot.json', 'r') as f:
    real_data = json.load(f)

# Calculate scaling factor
real_times = []
for family in real_data['families'].values():
    for task in family['tasks']:
        real_times.append(task['processing_time'])

real_avg = np.mean(real_times)
target_avg = 2.5  # Original synthetic data average
scale_factor = real_avg / target_avg

print(f"   Real average job duration: {real_avg:.1f}h")
print(f"   Target average: {target_avg}h")
print(f"   Scale factor: {scale_factor:.1f}x")

# Create scaled data
scaled_data = json.loads(json.dumps(real_data))  # Deep copy
for family in scaled_data['families'].values():
    for task in family['tasks']:
        # Scale down processing time
        task['processing_time'] = task['processing_time'] / scale_factor

# Save scaled data
scaled_path = 'data/scaled_production_snapshot.json'
with open(scaled_path, 'w') as f:
    json.dump(scaled_data, f, indent=2)

print(f"   Scaled data saved to: {scaled_path}")

# Calculate new workload
scaled_times = []
for family in scaled_data['families'].values():
    for task in family['tasks']:
        scaled_times.append(task['processing_time'])

scaled_total = sum(scaled_times)
print(f"   Scaled total workload: {scaled_total:.1f}h")
print(f"   Theoretical minimum: {scaled_total/149:.1f}h")

# Test both models
print("\n2. Testing models on scaled data...")

# Temporarily use scaled data
import shutil
shutil.copy('data/real_production_snapshot.json', 'data/real_production_snapshot.backup.json')
shutil.copy(scaled_path, 'data/real_production_snapshot.json')

models = [
    ("Original (1M steps)", "models/full_production/final_model.zip"),
    ("Extended (1.5M steps)", "models/full_production/extended/final_extended_model.zip")
]

results = {}

try:
    for model_name, model_path in models:
        if not os.path.exists(model_path):
            print(f"\n   {model_name}: Model not found")
            continue
            
        print(f"\n   Testing {model_name}...")
        model = PPO.load(model_path)
        
        # Create environment with scaled data
        env = FullProductionEnv(
            n_machines=150,
            n_jobs=500,
            state_compression="hierarchical",
            use_break_constraints=True,
            use_holiday_constraints=True,
            seed=42
        )
        
        # Run evaluation
        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        makespan = info.get('makespan', 0)
        completion = info.get('completion_rate', 1.0)
        
        # Scale makespan back up for real-world estimate
        real_makespan_estimate = makespan * scale_factor
        
        results[model_name] = {
            'scaled_makespan': makespan,
            'real_estimate': real_makespan_estimate,
            'completion': completion,
            'steps': steps
        }
        
        print(f"     Scaled makespan: {makespan:.1f}h")
        print(f"     Real estimate: {real_makespan_estimate:.1f}h")
        print(f"     Completion: {completion:.1%}")
        
        env.close()

finally:
    # Restore original data
    shutil.copy('data/real_production_snapshot.backup.json', 'data/real_production_snapshot.json')
    os.remove('data/real_production_snapshot.backup.json')

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"  Performance on scaled data: {result['scaled_makespan']:.1f}h")
    print(f"  Estimated real performance: {result['real_estimate']:.1f}h")
    print(f"  vs. Theoretical minimum: {result['real_estimate']/(scaled_total*scale_factor/149):.1f}x")

print(f"\nConclusion:")
if results:
    best_model = min(results.items(), key=lambda x: x[1]['real_estimate'])
    print(f"  Best model: {best_model[0]}")
    print(f"  Estimated real makespan: {best_model[1]['real_estimate']:.1f}h")
    print(f"  Recommendation: {'Viable for deployment' if best_model[1]['real_estimate'] < 150 else 'Needs retraining'}")
else:
    print("  No models tested successfully")