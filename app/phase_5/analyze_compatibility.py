#!/usr/bin/env python3
"""
Analyze compatibility matrix to understand valid actions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def analyze_compatibility():
    print("\n" + "="*60)
    print("Analyzing Compatibility Matrix")
    print("="*60 + "\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=411,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=100,
        seed=42
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Get compatibility matrix
    compat = env.compatibility_matrix
    print(f"Compatibility matrix shape: {compat.shape}")
    print(f"Total compatible pairs: {np.sum(compat)}")
    print(f"Percentage compatible: {np.sum(compat) / (compat.shape[0] * compat.shape[1]) * 100:.1f}%")
    
    # Analyze by job
    print("\n" + "-"*40)
    print("Jobs with NO compatible machines:")
    no_compat_jobs = []
    for j in range(compat.shape[0]):
        if np.sum(compat[j]) == 0:
            no_compat_jobs.append(j)
    
    print(f"Found {len(no_compat_jobs)} jobs with no compatible machines")
    if len(no_compat_jobs) > 0:
        print(f"Job indices: {no_compat_jobs[:20]}...")
        
        # Check specific job details
        if len(env.jobs) > no_compat_jobs[0]:
            job = env.jobs[no_compat_jobs[0]]
            print(f"\nExample job with no compatibility:")
            print(f"  Job ID: {job.get('job_id', 'Unknown')}")
            print(f"  Family ID: {job.get('family_id', 'Unknown')}")
            print(f"  Capable machines: {job.get('capable_machines', [])}")
    
    # Analyze by machine
    print("\n" + "-"*40)
    print("Machines that can process NO jobs:")
    no_job_machines = []
    for m in range(compat.shape[1]):
        if np.sum(compat[:, m]) == 0:
            no_job_machines.append(m)
    
    print(f"Found {len(no_job_machines)} machines that can't process any jobs")
    if len(no_job_machines) > 0:
        print(f"Machine indices: {no_job_machines[:20]}...")
    
    # Find some valid pairs
    print("\n" + "-"*40)
    print("Finding valid job-machine pairs:")
    valid_pairs = []
    for j in range(min(50, compat.shape[0])):  # Check first 50 jobs
        compatible_machines = np.where(compat[j])[0]
        if len(compatible_machines) > 0:
            valid_pairs.append((j, compatible_machines[0]))
            if len(valid_pairs) >= 10:
                break
    
    print(f"Found {len(valid_pairs)} valid pairs (showing first 10):")
    for job_idx, machine_idx in valid_pairs:
        job = env.jobs[job_idx] if job_idx < len(env.jobs) else {}
        machine = env.machines[machine_idx] if machine_idx < len(env.machines) else {}
        print(f"  Job {job_idx} ({job.get('job_id', 'Unknown')}) → Machine {machine_idx} ({machine.get('machine_name', 'Unknown')})")
    
    # Analyze specific problematic jobs
    print("\n" + "-"*40)
    print("Analyzing problematic job 333:")
    if 333 < compat.shape[0]:
        compatible_machines = np.where(compat[333])[0]
        print(f"Job 333 has {len(compatible_machines)} compatible machines")
        if len(compatible_machines) > 0:
            print(f"Compatible machine indices: {compatible_machines[:10].tolist()}")
        else:
            if 333 < len(env.jobs):
                job = env.jobs[333]
                print(f"Job details:")
                print(f"  Job ID: {job.get('job_id', 'Unknown')}")
                print(f"  Family ID: {job.get('family_id', 'Unknown')}")
                print(f"  Capable machines: {job.get('capable_machines', [])}")
    
    # Check action masks
    print("\n" + "-"*40)
    print("Checking action masks vs compatibility:")
    masks = env.get_action_masks()
    machine_masks = masks['machine']
    
    # Compare masks with compatibility
    discrepancies = 0
    for j in range(min(10, compat.shape[0])):
        compat_machines = np.sum(compat[j])
        mask_machines = np.sum(machine_masks[j]) if j < len(machine_masks) else 0
        
        if compat_machines != mask_machines:
            discrepancies += 1
            print(f"Job {j}: compat={compat_machines}, mask={mask_machines}")
    
    if discrepancies == 0:
        print("Action masks match compatibility matrix for first 10 jobs")

if __name__ == "__main__":
    analyze_compatibility()