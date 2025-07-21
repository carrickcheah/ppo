#!/usr/bin/env python3
"""
Test script for the real PPO scheduler integration.
"""

import json
import subprocess
import time

def test_real_scheduling():
    """Test the API with real PPO scheduling."""
    
    # Sample request with multiple jobs
    request_data = {
        "jobs": [
            {
                "job_id": "JOST25060240_CM17-002-1",
                "family_id": "JOST25060240",
                "sequence": 1,
                "processing_time": 2.5,
                "machine_types": [1, 2, 3],
                "is_important": True,
                "lcd_date": "2025-07-25T17:00:00",
                "setup_time": 0.3
            },
            {
                "job_id": "JOAW25060220_CP08-504A-1",
                "family_id": "JOAW25060220",
                "sequence": 1,
                "processing_time": 1.8,
                "machine_types": [2, 4, 5],
                "is_important": False,
                "lcd_date": "2025-07-28T17:00:00",
                "setup_time": 0.2
            },
            {
                "job_id": "JOST25060240_CM17-002-2",
                "family_id": "JOST25060240",
                "sequence": 2,
                "processing_time": 3.2,
                "machine_types": [1, 2, 3],
                "is_important": True,
                "lcd_date": "2025-07-25T17:00:00",
                "setup_time": 0.1
            },
            {
                "job_id": "JOTP25060180_CL02-789-1",
                "family_id": "JOTP25060180",
                "sequence": 1,
                "processing_time": 4.5,
                "machine_types": [5, 6, 7],
                "is_important": True,
                "lcd_date": "2025-07-24T12:00:00",
                "setup_time": 0.5
            },
            {
                "job_id": "JOEX25060150_AD02-123-1",
                "family_id": "JOEX25060150",
                "sequence": 1,
                "processing_time": 2.0,
                "machine_types": [1, 3, 5, 7],
                "is_important": False,
                "lcd_date": "2025-07-30T17:00:00",
                "setup_time": 0.3
            }
        ],
        "schedule_start": "2025-07-21T06:30:00",
        "respect_break_times": True,
        "respect_holidays": True
    }
    
    # Convert to JSON
    json_data = json.dumps(request_data, indent=2)
    
    # Test with curl
    print("Testing real PPO scheduling...")
    print(f"Sending {len(request_data['jobs'])} jobs for scheduling")
    print("-" * 60)
    
    cmd = [
        "curl", "-s", "-X", "POST",
        "http://localhost:8000/schedule",
        "-H", "Content-Type: application/json",
        "-H", "X-API-Key: dev-api-key-change-in-production",
        "-d", json_data
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            response = json.loads(result.stdout)
            
            print(f"✓ Schedule created: {response['schedule_id']}")
            print(f"✓ Algorithm used: {response['algorithm_used']}")
            print(f"✓ Generation time: {response['generation_time']:.3f} seconds")
            print(f"✓ Jobs scheduled: {response['metrics']['scheduled_jobs']}/{response['metrics']['total_jobs']}")
            print(f"✓ Makespan: {response['metrics']['makespan']:.1f} hours")
            print(f"✓ Completion rate: {response['metrics']['completion_rate']:.1f}%")
            print(f"✓ Average utilization: {response['metrics']['average_utilization']:.1f}%")
            
            if response.get('warnings'):
                print("\nWarnings:")
                for warning in response['warnings']:
                    print(f"  ⚠ {warning}")
            
            print("\nScheduled Jobs:")
            for job in response['scheduled_jobs'][:5]:  # Show first 5
                print(f"  - {job['job_id']} on {job['machine_name']} "
                      f"({job['start_time']:.1f}h - {job['end_time']:.1f}h)")
            
            if len(response['scheduled_jobs']) > 5:
                print(f"  ... and {len(response['scheduled_jobs']) - 5} more jobs")
                
        else:
            print(f"✗ Error: {result.stderr}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    # Wait a moment for server to be ready
    time.sleep(2)
    test_real_scheduling()