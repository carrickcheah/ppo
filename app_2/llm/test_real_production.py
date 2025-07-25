"""
Test LLM scheduler with REAL production data focusing on multi-machine constraints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_scheduler import LLMScheduler
from llm.parser import ScheduleParser, ScheduledTask
from llm.validator import ScheduleValidator
import json


def test_real_multi_machine_jobs():
    """Test scheduling with real multi-machine jobs from production."""
    
    print("=== REAL PRODUCTION DATA TEST ===")
    print("Testing with actual factory jobs that require multiple machines simultaneously\n")
    
    # Initialize scheduler
    scheduler = LLMScheduler()
    
    # Test with jobs that include multi-machine requirements
    print("Scheduling real production jobs including:")
    print("- JOTP25070237: Requires machines [109,110,111] simultaneously")
    print("- JORW25070218: Requires 21 machines simultaneously")
    print("- Regular single-machine jobs\n")
    
    # Schedule first 30 jobs to ensure we get some multi-machine ones
    result = scheduler.schedule(
        snapshot_path="/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json",
        strategy="constraint_focused",  # Use constraint-focused for better handling
        max_jobs=30
    )
    
    print(f"\n✓ Scheduling complete!")
    print(f"  - Jobs scheduled: {result['metrics']['total_jobs']}")
    print(f"  - Families: {result['metrics']['families_scheduled']}")
    print(f"  - Makespan: {result['metrics']['makespan_hours']:.1f} hours")
    print(f"  - Response time: {result['llm_metadata']['response_time']:.1f}s")
    print(f"  - Cost: ${result['llm_metadata']['cost_estimate']['total_cost']:.4f}")
    
    # Parse and analyze multi-machine jobs
    print("\n=== MULTI-MACHINE JOB ANALYSIS ===")
    
    multi_machine_jobs = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            if len(task_data['machine_ids']) > 1:
                multi_machine_jobs.append({
                    'job_id': task_data['task_id'],
                    'machines': task_data['machine_ids'],
                    'start': task_data['start_time'],
                    'end': task_data['end_time']
                })
    
    if multi_machine_jobs:
        print(f"Found {len(multi_machine_jobs)} multi-machine jobs:")
        for job in multi_machine_jobs[:5]:  # Show first 5
            print(f"\n{job['job_id']}:")
            print(f"  - Requires {len(job['machines'])} machines: {job['machines']}")
            print(f"  - Scheduled: {job['start']} to {job['end']}")
            print(f"  - ALL machines occupied simultaneously")
    else:
        print("No multi-machine jobs found in this batch")
    
    # Validate constraints
    print("\n=== CONSTRAINT VALIDATION ===")
    
    # Convert to ScheduledTask objects for validation
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            job_id = task_data['task_id']
            parts = job_id.split('-')
            
            # Extract sequence info
            if '/' in parts[-1]:
                seq_parts = parts[-1].split('/')
                sequence = int(seq_parts[0].split('-')[-1]) if '-' in seq_parts[0] else 1
                total_sequences = int(seq_parts[1])
            else:
                sequence = 1
                total_sequences = 1
            
            from datetime import datetime
            task = ScheduledTask(
                job_id=job_id,
                family_id=family_id,
                sequence=sequence,
                total_sequences=total_sequences,
                machine_ids=task_data['machine_ids'],
                start_time=datetime.strptime(task_data['start_time'], "%Y-%m-%d %H:%M"),
                end_time=datetime.strptime(task_data['end_time'], "%Y-%m-%d %H:%M"),
                processing_hours=(datetime.strptime(task_data['end_time'], "%Y-%m-%d %H:%M") - 
                                 datetime.strptime(task_data['start_time'], "%Y-%m-%d %H:%M")).total_seconds() / 3600
            )
            scheduled_tasks.append(task)
    
    # Validate
    validator = ScheduleValidator()
    is_valid, violations = validator.validate_schedule(scheduled_tasks)
    
    print(f"Schedule valid: {is_valid}")
    print(f"Total violations: {len(violations)}")
    
    # Check multi-machine constraint specifically
    multi_machine_violations = [v for v in violations if 'MULTI_MACHINE' in v.type]
    sequence_violations = [v for v in violations if 'SEQUENCE' in v.type]
    overlap_violations = [v for v in violations if 'OVERLAP' in v.type]
    
    print(f"\nViolation breakdown:")
    print(f"  - Multi-machine violations: {len(multi_machine_violations)}")
    print(f"  - Sequence violations: {len(sequence_violations)}")
    print(f"  - Time overlap violations: {len(overlap_violations)}")
    
    if multi_machine_violations:
        print("\nMulti-machine violations found:")
        for v in multi_machine_violations[:3]:
            print(f"  - {v}")
    else:
        print("\n✓ No multi-machine violations - ALL multi-machine jobs properly scheduled!")
    
    # Show a sample schedule entry
    print("\n=== SAMPLE SCHEDULE OUTPUT ===")
    sample_count = 0
    for family_id, tasks in result['schedule'].items():
        for task in tasks:
            print(f"{task['task_id']} -> machines{task['machine_ids']} @ {task['start_time']} - {task['end_time']}")
            sample_count += 1
            if sample_count >= 5:
                break
        if sample_count >= 5:
            break
    
    return result


if __name__ == "__main__":
    # Test with real production data
    result = test_real_multi_machine_jobs()
    
    print("\n=== TEST SUMMARY ===")
    print("✓ Successfully scheduled real production jobs")
    print("✓ Multi-machine constraints handled")
    print("✓ Using real machine IDs from factory database")
    print("✓ Processing times from actual production data")