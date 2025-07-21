#!/usr/bin/env python3
"""
Debug why environment terminates after 172 steps
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.full_production_env import FullProductionEnv
import numpy as np

print("DEBUGGING ENVIRONMENT TERMINATION")
print("="*40)

# Create environment
env = FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)

print(f"\nEnvironment configuration:")
print(f"- Total machines: {len(env.machines)}")
print(f"- Total jobs: {len(env.jobs)}")
print(f"- Max episode steps: {env.max_episode_steps}")
print(f"- Max valid actions: {env.max_valid_actions}")

# Reset and check initial state
obs, info = env.reset()
print(f"\nInitial state:")
print(f"- Observation shape: {obs.shape}")
print(f"- Valid actions available: {len(env.valid_actions)}")

# Take random actions and monitor
print(f"\nTaking random actions...")
for step in range(200):
    if not env.valid_actions:
        print(f"\nNo valid actions at step {step}")
        break
        
    # Random action
    action = np.random.randint(0, env.action_space.n)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 50 == 0 or terminated or truncated:
        scheduled = sum(len(tasks) for tasks in env.completed_tasks.values()) if hasattr(env, 'completed_tasks') else 0
        print(f"Step {step}: Valid actions={len(env.valid_actions)}, Scheduled={scheduled}, Done={terminated or truncated}")
    
    if terminated or truncated:
        print(f"\nEpisode ended at step {step + 1}")
        print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
        print(f"Jobs scheduled: {scheduled}/{len(env.jobs)}")
        print(f"Makespan: {info.get('makespan', 0):.1f}h")
        break

# Check what's limiting the actions
print(f"\nAnalyzing action space:")
print(f"- Action space size: {env.action_space.n}")
print(f"- Max valid actions parameter: {env.max_valid_actions}")

# Check if all jobs can be scheduled
total_job_machine_pairs = 0
for job in env.jobs[:10]:  # Sample first 10 jobs
    allowed_machines = job.get('allowed_machine_types', [])
    compatible_machines = sum(1 for m in env.machines if m['machine_type_id'] in allowed_machines)
    total_job_machine_pairs += compatible_machines
    print(f"Job {job['job_id']}: {compatible_machines} compatible machines")

print(f"\nPotential issue identified:")
if env.max_valid_actions < len(env.jobs):
    print(f"⚠️  max_valid_actions ({env.max_valid_actions}) < total jobs ({len(env.jobs)})")
    print("   This limits how many actions can be presented per step!")

env.close()