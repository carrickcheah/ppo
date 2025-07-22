#!/usr/bin/env python3
"""
Debug machine ID mapping issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import json
from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def debug_machine_mapping():
    """Debug the machine ID mapping."""
    print("\n" + "="*60)
    print("Debugging Machine ID Mapping")
    print("="*60 + "\n")
    
    # Load raw data to check
    with open("data/real_production_snapshot.json", 'r') as f:
        data = json.load(f)
    
    # Check first job's capable machines
    families = data['families']
    first_family = list(families.values())[0]
    first_task = first_family['tasks'][0]
    
    print("1. Raw Data Check:")
    print(f"   First job ID: {first_family['job_reference']}")
    print(f"   First task capable machines: {first_task['capable_machines'][:10]}...")
    
    # Check machine data
    machines = data['machines']
    print(f"\n2. Machine Data:")
    print(f"   Total machines in data: {len(machines)}")
    if isinstance(machines, list):
        print(f"   First 5 machines: {[m.get('machine_id', 'No ID') for m in machines[:5]]}")
    else:
        print(f"   First 5 machine IDs: {list(machines.keys())[:5]}")
    
    # Create environment and check
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=100,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=42
    )
    
    env.reset()
    
    print(f"\n3. Environment Machine Mapping:")
    print(f"   Number of machines: {len(env.machines)}")
    
    # Check first few machines
    for i in range(min(5, len(env.machines))):
        machine = env.machines[i]
        if isinstance(machine, dict):
            m_id = machine.get('machine_id', 'No ID')
            m_name = machine.get('name', 'No name')
        else:
            m_id = getattr(machine, 'machine_id', 'No ID')
            m_name = getattr(machine, 'name', 'No name')
        print(f"   Machine[{i}]: ID={m_id}, Name={m_name}")
    
    print(f"\n4. Job Capability Check:")
    # Check first job's capable machines
    if len(env.jobs) > 0:
        first_job = env.jobs[0]
        if isinstance(first_job, dict):
            capable = first_job.get('capable_machines', [])
        else:
            capable = getattr(first_job, 'capable_machines', [])
        print(f"   First job capable machines: {capable[:10] if capable else 'None'}...")
        
        # Map machine IDs to indices
        print(f"\n5. Machine ID to Index Mapping:")
        machine_id_to_idx = {}
        for idx, machine in enumerate(env.machines):
            if isinstance(machine, dict):
                m_id = machine.get('machine_id', idx)
            else:
                m_id = getattr(machine, 'machine_id', idx)
            machine_id_to_idx[m_id] = idx
        
        print(f"   Created mapping for {len(machine_id_to_idx)} machines")
        
        # Check if capable machine IDs exist in our mapping
        if capable:
            found = 0
            for cap_id in capable[:5]:
                if cap_id in machine_id_to_idx:
                    found += 1
                    print(f"   Machine ID {cap_id} -> Index {machine_id_to_idx[cap_id]}")
                else:
                    print(f"   Machine ID {cap_id} -> NOT FOUND in environment")
            print(f"   Found {found}/{len(capable[:5])} capable machines in environment")

if __name__ == "__main__":
    debug_machine_mapping()