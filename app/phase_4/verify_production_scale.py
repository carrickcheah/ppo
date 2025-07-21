#!/usr/bin/env python3
"""
Verify the scale of production data being used
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.full_production_env import FullProductionEnv
import json

print("VERIFYING PRODUCTION SCALE")
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

# Check what data was loaded
print(f"\nEnvironment Statistics:")
print(f"- Machines available: {len(env.machines)}")
print(f"- Jobs loaded: {len(env.jobs)}")
print(f"- Families loaded: {len(env.families)}")

# Show sample jobs
print(f"\nSample Jobs (first 5):")
for i, job in enumerate(env.jobs[:5]):
    print(f"  {i+1}. {job['workorder']} - {job['processing_time']:.1f}h - Priority {job['priority']}")

# Calculate total processing time
total_processing = sum(job['processing_time'] for job in env.jobs)
theoretical_min = total_processing / len(env.machines)

print(f"\nWorkload Analysis:")
print(f"- Total processing time: {total_processing:.1f}h")
print(f"- Theoretical minimum makespan: {theoretical_min:.1f}h")
print(f"- Average job duration: {total_processing/len(env.jobs):.2f}h")

# Check data source
data_path = env.data_file
print(f"\nData source: {data_path}")

# Load and check snapshot
with open(data_path, 'r') as f:
    snapshot = json.load(f)
    
print(f"\nSnapshot metadata:")
for key, value in snapshot['metadata'].items():
    print(f"  {key}: {value}")

env.close()