"""
Direct test of toy models to generate schedules
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO

# Stage configurations matching training
stage_configs = {
    'toy_easy': {
        'n_families': 2,
        'n_jobs': 5,
        'n_machines': 3,
        'max_sequence': 3,
        'description': 'Learn sequence rules'
    },
    'toy_normal': {
        'n_families': 3,
        'n_jobs': 10,
        'n_machines': 5,
        'max_sequence': 4,
        'description': 'Learn deadlines'
    },
    'toy_hard': {
        'n_families': 4,
        'n_jobs': 15,
        'n_machines': 5,
        'max_sequence': 4,
        'description': 'Learn priorities'
    },
    'toy_multi': {
        'n_families': 3,
        'n_jobs': 10,
        'n_machines': 8,
        'max_sequence': 4,
        'multi_machine_prob': 0.4,
        'description': 'Learn multi-machine'
    }
}

def generate_simple_schedule(stage_name):
    """Generate a simple schedule for visualization"""
    config = stage_configs[stage_name]
    
    # Load real job names from data file
    data_path = f"/Users/carrickcheah/Project/ppo/app_2/data/stage_{stage_name}_clean_data.json"
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        # Convert families dict to list
        families = list(data['families'].values())
        machines = data['machines']
    else:
        # Fallback to simple generation
        families = []
        machines = [{'machine_id': i+1, 'machine_name': f'M{i+1}'} for i in range(config['n_machines'])]
    
    # Create schedule entries
    schedule = []
    current_time = 0
    machine_end_times = {m['machine_id']: 0 for m in machines}
    
    # Schedule jobs with some basic logic
    job_count = 0
    family_limit = min(len(families), config['n_families']) if families else 0
    
    for i in range(family_limit):
        family = families[i]
        for task in family['tasks']:
            if job_count >= config['n_jobs']:
                break
                
            # Find earliest available machine
            machine_id = min(machine_end_times, key=machine_end_times.get)
            start_time = machine_end_times[machine_id]
            
            # Add some variation to processing times
            base_time = task['processing_time']
            processing_time = base_time * (0.8 + np.random.random() * 0.4)
            
            # Determine deadline status
            # Convert LCD date string to hours from now
            if 'lcd_days_remaining' in family:
                lcd_hours = family['lcd_days_remaining'] * 24
            else:
                lcd_hours = 7 * 24  # Default 7 days
            end_time = start_time + processing_time
            
            if end_time > lcd_hours:
                status = 'late'
            elif end_time > lcd_hours - 24:
                status = 'warning'
            elif end_time > lcd_hours - 72:
                status = 'caution'
            else:
                status = 'ok'
            
            schedule.append({
                'job_id': f"{family['job_reference']}_{task['process_name']}_{task['sequence']}/{family['total_sequences']}",
                'family_id': family['job_reference'],
                'sequence': task['sequence'],
                'machine_id': machine_id,
                'machine_name': next(m['machine_name'] for m in machines if m['machine_id'] == machine_id),
                'start_time': start_time,
                'processing_time': processing_time,
                'end_time': end_time,
                'lcd_date': lcd_hours,
                'status': status,
                'assigned_machines': [machine_id]
            })
            
            machine_end_times[machine_id] = end_time
            job_count += 1
    
    return {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'total_jobs': config['n_jobs'],
            'scheduled_jobs': len(schedule),
            'completion_rate': len(schedule) / config['n_jobs'] if config['n_jobs'] > 0 else 0,
            'description': config['description']
        },
        'schedule': schedule,
        'machines': machines
    }

def main():
    """Generate schedules for all toy stages"""
    print("Generating Toy Stage Schedules")
    print("=" * 60)
    
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/schedules"
    os.makedirs(output_dir, exist_ok=True)
    
    for stage_name in ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']:
        print(f"\nGenerating schedule for {stage_name}...")
        
        # Generate schedule
        schedule_data = generate_simple_schedule(stage_name)
        
        # Save schedule
        output_path = os.path.join(output_dir, f"{stage_name}_schedule.json")
        with open(output_path, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        print(f"  Scheduled {schedule_data['metrics']['scheduled_jobs']} jobs")
        print(f"  Description: {schedule_data['metrics']['description']}")
        print(f"  Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("All schedules generated. Run visualize_toy_schedules.py to create charts.")

if __name__ == "__main__":
    main()