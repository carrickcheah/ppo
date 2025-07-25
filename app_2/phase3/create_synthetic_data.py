"""
Create synthetic data for Phase 3 training
Focus on toy stages first
"""

import json
import os
import random
from datetime import datetime, timedelta


def create_synthetic_snapshot(n_jobs, n_machines, output_path, multi_machine_ratio=0.1):
    """Create a synthetic data snapshot for testing."""
    
    # Create machines
    machines = []
    for i in range(n_machines):
        machines.append({
            'machine_id': i + 1,
            'machine_name': f'M{i+1:02d}',
            'machine_type_id': (i % 3) + 1  # 3 types of machines
        })
    
    # Create job families
    families = {}
    current_time = datetime.now()
    
    for i in range(n_jobs):
        family_id = f'JOB_{i:04d}'
        
        # Random properties
        is_important = random.random() < 0.3  # 30% important
        lcd_days = random.randint(1, 14) if is_important else random.randint(3, 21)
        n_sequences = random.randint(1, 4)  # 1-4 sequences per job
        
        # Create tasks (sequences)
        tasks = []
        for seq in range(1, n_sequences + 1):
            # Processing time in hours
            processing_time = random.uniform(0.5, 8.0)
            
            # Capable machines
            if random.random() < multi_machine_ratio and n_machines > 3:
                # Multi-machine task
                n_capable = random.randint(2, min(5, n_machines))
                capable_machines = random.sample(range(1, n_machines + 1), n_capable)
            else:
                # Single machine task
                # Prefer machines of certain types for variety
                if seq == 1:
                    # First sequence prefers type 1 machines
                    capable_machines = [m['machine_id'] for m in machines if m['machine_type_id'] == 1]
                elif seq == n_sequences:
                    # Last sequence prefers type 3 machines
                    capable_machines = [m['machine_id'] for m in machines if m['machine_type_id'] == 3]
                else:
                    # Middle sequences prefer type 2 machines
                    capable_machines = [m['machine_id'] for m in machines if m['machine_type_id'] == 2]
                
                # If no machines of preferred type, allow any
                if not capable_machines:
                    capable_machines = [random.randint(1, n_machines)]
            
            tasks.append({
                'sequence': seq,
                'process_name': f'PROCESS_{seq}',
                'processing_time': round(processing_time, 2),
                'capable_machines': capable_machines,
                'status': 'pending'
            })
        
        families[family_id] = {
            'job_reference': family_id,
            'product': f'PRODUCT_{i % 10}',  # 10 product types
            'is_important': is_important,
            'lcd_days_remaining': lcd_days,
            'total_sequences': n_sequences,
            'tasks': tasks,
            'customer': f'CUSTOMER_{i % 5}'  # 5 customers
        }
    
    # Create snapshot
    snapshot = {
        'families': families,
        'machines': machines,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'snapshot_type': 'synthetic',
            'total_families': len(families),
            'total_tasks': sum(len(f['tasks']) for f in families.values()),
            'total_machines': len(machines)
        }
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    return snapshot


def create_all_synthetic_snapshots():
    """Create synthetic snapshots for all toy stages."""
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("=== CREATING SYNTHETIC DATA FOR PHASE 3 ===\n")
    
    # Toy stages configurations
    toy_configs = [
        ('toy_easy', 5, 3, 0.0),  # No multi-machine
        ('toy_normal', 10, 5, 0.0),
        ('toy_hard', 15, 5, 0.0),
        ('toy_multi', 10, 8, 0.3),  # 30% multi-machine
    ]
    
    # Small stages
    small_configs = [
        ('small_balanced', 30, 15, 0.1),
        ('small_rush', 50, 20, 0.1),
        ('small_bottleneck', 40, 10, 0.05),
        ('small_complex', 50, 25, 0.2),
    ]
    
    # Create toy snapshots
    print("Creating Toy Stage Snapshots:")
    for name, n_jobs, n_machines, multi_ratio in toy_configs:
        path = os.path.join(data_dir, f'snapshot_{name}.json')
        snapshot = create_synthetic_snapshot(n_jobs, n_machines, path, multi_ratio)
        
        n_multi = sum(1 for f in snapshot['families'].values() 
                     for t in f['tasks'] 
                     if len(t['capable_machines']) > 1)
        
        print(f"  ✓ {name}: {n_jobs} jobs, {n_machines} machines, {n_multi} multi-machine tasks")
    
    print("\nCreating Small Stage Snapshots:")
    for name, n_jobs, n_machines, multi_ratio in small_configs:
        path = os.path.join(data_dir, f'snapshot_{name}.json')
        snapshot = create_synthetic_snapshot(n_jobs, n_machines, path, multi_ratio)
        
        n_multi = sum(1 for f in snapshot['families'].values() 
                     for t in f['tasks'] 
                     if len(t['capable_machines']) > 1)
        
        print(f"  ✓ {name}: {n_jobs} jobs, {n_machines} machines, {n_multi} multi-machine tasks")
    
    # Create a special rush snapshot with urgent deadlines
    print("\nCreating Special Rush Snapshot:")
    rush_path = os.path.join(data_dir, 'snapshot_rush.json')
    rush_snapshot = create_synthetic_snapshot(50, 20, rush_path, 0.1)
    
    # Make all jobs urgent
    for family in rush_snapshot['families'].values():
        family['lcd_days_remaining'] = random.randint(1, 3)  # 1-3 days only
        family['is_important'] = random.random() < 0.5  # 50% important
    
    with open(rush_path, 'w') as f:
        json.dump(rush_snapshot, f, indent=2)
    
    print("  ✓ rush: 50 jobs with urgent deadlines (1-3 days)")
    
    print("\n✓ Synthetic data creation complete!")
    print(f"\nAll snapshots saved in: {data_dir}")


if __name__ == "__main__":
    create_all_synthetic_snapshots()