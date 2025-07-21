#!/usr/bin/env python3
"""
Unit test for the scheduler module to debug issues.
"""

import sys
sys.path.append('/Users/carrickcheah/Project/ppo/app')

from datetime import datetime
from src.deployment.models import Job, Machine
from src.deployment.scheduler import MockScheduler

def test_mock_scheduler():
    """Test the mock scheduler directly."""
    print("Testing Mock Scheduler...")
    
    # Create test data
    jobs = [
        Job(
            job_id="TEST001",
            family_id="FAM001",
            sequence=1,
            processing_time=2.0,
            machine_types=[1, 2],
            is_important=True,
            lcd_date=datetime(2025, 7, 25, 17, 0)
        ),
        Job(
            job_id="TEST002",
            family_id="FAM002",
            sequence=1,
            processing_time=3.0,
            machine_types=[2, 3],
            is_important=False,
            lcd_date=datetime(2025, 7, 28, 17, 0)
        )
    ]
    
    machines = [
        Machine(machine_id=1, machine_name="M001", machine_type=1),
        Machine(machine_id=2, machine_name="M002", machine_type=2),
        Machine(machine_id=3, machine_name="M003", machine_type=3)
    ]
    
    # Create scheduler
    scheduler = MockScheduler()
    
    # Test scheduling
    try:
        scheduled_jobs, metrics = scheduler.schedule(
            jobs=jobs,
            machines=machines,
            schedule_start=datetime(2025, 7, 21, 6, 30)
        )
        
        print(f"✓ Scheduled {len(scheduled_jobs)} jobs")
        print(f"✓ Makespan: {metrics['makespan']} hours")
        print(f"✓ Completion rate: {metrics['completion_rate']}%")
        
        for job in scheduled_jobs:
            print(f"  - {job.job_id} on {job.machine_name} at {job.start_time:.1f}h")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_mock_scheduler()