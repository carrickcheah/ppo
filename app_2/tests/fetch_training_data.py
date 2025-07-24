#!/usr/bin/env python3
"""
Fetch training data from database in the exact format needed for model training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.db_connector import DBConnector
import json
from datetime import datetime

def fetch_training_data():
    """Fetch and display training data in the required format."""
    print("Fetching Training Data from Database...")
    print("=" * 80)
    
    # Create connector
    db = DBConnector()
    
    try:
        # Connect
        db.connect()
        
        # Fetch jobs
        print("\n1. FETCHING JOBS...")
        jobs = db.fetch_pending_jobs()
        
        # Transform to required format
        formatted_jobs = []
        for job in jobs:
            # Note: we now have 'required_machines' not 'machine_types'
            formatted_job = {
                'job_id': job['job_id'],
                'family_id': job['family_id'],
                'sequence': job['sequence'],
                'required_machines': job['required_machines'],  # Machine IDs that must ALL be used
                'processing_time': round(job['processing_time'], 2),
                'lcd_days_remaining': job['lcd_days_remaining'],
                'is_important': int(job['is_important'])
            }
            formatted_jobs.append(formatted_job)
        
        # Fetch machines
        print("\n2. FETCHING MACHINES...")
        machines = db.fetch_machines()
        
        # Transform to required format
        formatted_machines = []
        for machine in machines:
            formatted_machine = {
                'machine_id': machine['machine_id'],  # 0-based index
                'machine_name': machine['machine_name'],
                'machine_type_id': machine['machine_type_id'],
                'db_machine_id': machine['db_machine_id']  # Original DB ID
            }
            formatted_machines.append(formatted_machine)
        
        # Display sample data
        print("\n" + "=" * 80)
        print("SAMPLE JOB DATA (First 5 jobs):")
        print("=" * 80)
        print("# Each job has:")
        for job in formatted_jobs[:5]:
            print(json.dumps(job, indent=4))
            print()
        
        print("\n" + "=" * 80)
        print("SAMPLE MACHINE DATA (First 5 machines):")
        print("=" * 80)
        print("# Each machine has:")
        for machine in formatted_machines[:5]:
            print(json.dumps(machine, indent=4))
            print()
        
        # Show statistics
        print("\n" + "=" * 80)
        print("DATA STATISTICS:")
        print("=" * 80)
        print(f"Total Jobs: {len(formatted_jobs)}")
        print(f"Total Machines: {len(formatted_machines)}")
        
        # Multi-machine jobs
        multi_machine_jobs = [j for j in formatted_jobs if len(j['required_machines']) > 1]
        print(f"Multi-Machine Jobs: {len(multi_machine_jobs)}")
        
        # Important jobs
        important_jobs = [j for j in formatted_jobs if j['is_important'] == 1]
        print(f"Important Jobs: {len(important_jobs)}")
        
        # Processing time distribution
        processing_times = [j['processing_time'] for j in formatted_jobs]
        print(f"Processing Time Range: {min(processing_times):.2f} - {max(processing_times):.2f} hours")
        print(f"Average Processing Time: {sum(processing_times)/len(processing_times):.2f} hours")
        
        # Save to files for training
        print("\n" + "=" * 80)
        print("SAVING DATA FOR TRAINING:")
        print("=" * 80)
        
        # Create data directory
        os.makedirs('data/snapshots', exist_ok=True)
        
        # Save jobs
        jobs_file = f'data/snapshots/jobs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(jobs_file, 'w') as f:
            json.dump(formatted_jobs, f, indent=2)
        print(f"✓ Jobs saved to: {jobs_file}")
        
        # Save machines
        machines_file = f'data/snapshots/machines_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(machines_file, 'w') as f:
            json.dump(formatted_machines, f, indent=2)
        print(f"✓ Machines saved to: {machines_file}")
        
        # Create combined snapshot
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'jobs': formatted_jobs,
            'machines': formatted_machines,
            'statistics': {
                'total_jobs': len(formatted_jobs),
                'total_machines': len(formatted_machines),
                'multi_machine_jobs': len(multi_machine_jobs),
                'important_jobs': len(important_jobs),
                'processing_time_range': [min(processing_times), max(processing_times)],
                'average_processing_time': sum(processing_times)/len(processing_times)
            }
        }
        
        snapshot_file = f'data/snapshots/training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        print(f"✓ Complete snapshot saved to: {snapshot_file}")
        
        # Show example of multi-machine job
        if multi_machine_jobs:
            print("\n" + "=" * 80)
            print("EXAMPLE MULTI-MACHINE JOB:")
            print("=" * 80)
            example = multi_machine_jobs[0]
            print(f"Job {example['job_id']} requires {len(example['required_machines'])} machines simultaneously:")
            print(f"Machines needed: {example['required_machines']}")
            print(f"Processing time: {example['processing_time']} hours")
            print("This means ALL these machines will be occupied for the entire duration!")
        
        return formatted_jobs, formatted_machines
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        db.disconnect()
        print("\n✓ Database connection closed.")

if __name__ == "__main__":
    fetch_training_data()