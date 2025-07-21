#!/usr/bin/env python3
"""
Final evaluation of Phase 4 extended model
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

print("="*60)
print("PHASE 4 FINAL EVALUATION")
print("="*60)

# Load extended model
model_path = "models/full_production/extended/final_extended_model.zip"
print(f"\nLoading model: {model_path}")

if not os.path.exists(model_path):
    print("Extended model not found! Using original model.")
    model_path = "models/full_production/final_model.zip"

model = PPO.load(model_path)

# Create environment
env = FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)

print(f"\nEnvironment loaded:")
print(f"- Machines: {len(env.machines)}")
print(f"- Jobs: {len(env.jobs)}")
print(f"- Total workload: {sum(j['processing_time'] for j in env.jobs):.1f}h")

# Run single evaluation
print(f"\nRunning evaluation...")
obs, info = env.reset()
terminated = False
truncated = False
steps = 0

while not (terminated or truncated) and steps < 2000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    
    if steps % 100 == 0:
        print(f"  Step {steps}...")

makespan = info.get('makespan', 0)
scheduled = info.get('n_jobs_scheduled', 0)
completion = scheduled / len(env.jobs) if env.jobs else 0

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Makespan: {makespan:.1f}h")
print(f"Jobs scheduled: {scheduled}/{len(env.jobs)}")
print(f"Completion rate: {completion:.1%}")
print(f"Steps taken: {steps}")

if makespan < 45:
    print(f"\n✅ PHASE 4 COMPLETE! Target <45h achieved: {makespan:.1f}h")
else:
    print(f"\n⚠️  Current: {makespan:.1f}h, Target: <45h")

env.close()