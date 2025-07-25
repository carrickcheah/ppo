"""
Phase 3 Data Preparation
Create multiple data snapshots for curriculum learning
"""

import os
import subprocess
import json
import numpy as np
from datetime import datetime


def create_all_snapshots():
    """Create all data snapshots needed for Phase 3 training."""
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("=== PHASE 3 DATA PREPARATION ===\n")
    
    # Path to ingest_data.py
    ingest_script = "/Users/carrickcheah/Project/ppo/app/src/data_ingestion/ingest_data.py"
    
    # 1. Normal load snapshot
    print("1. Creating normal load snapshot...")
    normal_path = os.path.join(data_dir, 'snapshot_normal.json')
    
    # Call ingest_data.py with appropriate parameters
    cmd = [
        "python", ingest_script,
        "--output", normal_path,
        "--horizon", "7"  # 7 days ahead
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✓ Saved to: {normal_path}")
    else:
        print(f"   ✗ Error: {result.stderr}")
        return
    
    # 2. Rush orders snapshot (urgent deadlines)
    print("\n2. Creating rush orders snapshot...")
    rush_path = os.path.join(data_dir, 'snapshot_rush.json')
    # For rush orders, we'll post-process the normal snapshot
    with open(normal_path, 'r') as f:
        rush_data = json.load(f)
    
    # Make 80% of jobs urgent (reduce LCD times)
    for family_id, family in rush_data['families'].items():
        if np.random.random() < 0.8:  # 80% chance
            # Reduce LCD to 1-3 days
            family['lcd_days_remaining'] = np.random.randint(1, 4)
            family['is_important'] = np.random.random() < 0.5  # 50% also important
    
    with open(rush_path, 'w') as f:
        json.dump(rush_data, f, indent=2)
    print(f"   ✓ Saved to: {rush_path}")
    
    # 3. Heavy load snapshot (500+ jobs)
    print("\n3. Creating heavy load snapshot...")
    heavy_path = os.path.join(data_dir, 'snapshot_heavy.json')
    create_production_snapshot(
        output_file=heavy_path,
        days_ahead=14,  # Look further ahead for more jobs
        include_completed=False
    )
    
    # If we don't have enough jobs, we'll duplicate some
    with open(heavy_path, 'r') as f:
        heavy_data = json.load(f)
    
    original_count = len(heavy_data['families'])
    if original_count < 500:
        print(f"   Only {original_count} jobs found, duplicating to reach 500+...")
        
        # Duplicate jobs with slight variations
        new_families = {}
        counter = 0
        
        while len(heavy_data['families']) + len(new_families) < 500:
            for family_id, family in list(heavy_data['families'].items()):
                if len(heavy_data['families']) + len(new_families) >= 500:
                    break
                    
                # Create variation
                new_id = f"{family_id}_DUP{counter}"
                new_family = family.copy()
                new_family['job_reference'] = new_id
                new_family['lcd_days_remaining'] = max(1, family['lcd_days_remaining'] + np.random.randint(-2, 3))
                new_family['is_important'] = np.random.random() < 0.3
                
                # Copy tasks with slight time variations
                new_family['tasks'] = []
                for task in family['tasks']:
                    new_task = task.copy()
                    new_task['processing_time'] *= np.random.uniform(0.8, 1.2)
                    new_family['tasks'].append(new_task)
                
                new_families[new_id] = new_family
                counter += 1
        
        heavy_data['families'].update(new_families)
        heavy_data['metadata']['total_families'] = len(heavy_data['families'])
    
    with open(heavy_path, 'w') as f:
        json.dump(heavy_data, f, indent=2)
    print(f"   ✓ Saved to: {heavy_path} ({len(heavy_data['families'])} jobs)")
    
    # 4. Multi-machine heavy snapshot (30% multi-machine jobs)
    print("\n4. Creating multi-machine heavy snapshot...")
    multi_path = os.path.join(data_dir, 'snapshot_multi.json')
    
    # Start with normal snapshot
    with open(normal_path, 'r') as f:
        multi_data = json.load(f)
    
    # Make 30% of tasks require multiple machines
    for family in multi_data['families'].values():
        for task in family['tasks']:
            if np.random.random() < 0.3:  # 30% chance
                # Get current capable machines
                if isinstance(task['capable_machines'], list):
                    current_machines = task['capable_machines']
                else:
                    current_machines = [task['capable_machines']]
                
                # Add 1-3 more machines
                all_machine_ids = list(range(1, len(multi_data['machines']) + 1))
                additional = np.random.choice(
                    [m for m in all_machine_ids if m not in current_machines],
                    size=min(np.random.randint(1, 4), len(all_machine_ids) - len(current_machines)),
                    replace=False
                ).tolist()
                
                task['capable_machines'] = current_machines + additional
                task['is_multi_machine'] = True
    
    with open(multi_path, 'w') as f:
        json.dump(multi_data, f, indent=2)
    print(f"   ✓ Saved to: {multi_path}")
    
    # 5. Create synthetic data for toy stages
    print("\n5. Creating synthetic toy data...")
    
    # This will be handled by the curriculum environment itself
    # as it needs to generate fresh data for each episode
    
    print("\n✓ Data preparation complete!")
    print(f"\nSnapshots created in: {data_dir}")
    print("  - snapshot_normal.json")
    print("  - snapshot_rush.json")
    print("  - snapshot_heavy.json")
    print("  - snapshot_multi.json")
    
    # Print summary statistics
    print("\nSnapshot Statistics:")
    for snapshot_name in ['normal', 'rush', 'heavy', 'multi']:
        path = os.path.join(data_dir, f'snapshot_{snapshot_name}.json')
        with open(path, 'r') as f:
            data = json.load(f)
        
        n_jobs = len(data['families'])
        n_tasks = sum(len(f['tasks']) for f in data['families'].values())
        n_machines = len(data['machines'])
        n_important = sum(1 for f in data['families'].values() if f.get('is_important', False))
        n_multi = sum(1 for f in data['families'].values() 
                     for t in f['tasks'] 
                     if len(t.get('capable_machines', [])) > 1)
        
        print(f"\n  {snapshot_name}:")
        print(f"    Jobs: {n_jobs}")
        print(f"    Tasks: {n_tasks}")
        print(f"    Machines: {n_machines}")
        print(f"    Important: {n_important} ({n_important/n_jobs*100:.1f}%)")
        print(f"    Multi-machine tasks: {n_multi} ({n_multi/n_tasks*100:.1f}%)")


if __name__ == "__main__":
    create_all_snapshots()