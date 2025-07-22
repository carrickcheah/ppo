import requests
import json
from datetime import datetime, timedelta

# Test job data
jobs = []
for i in range(20):
    job = {
        "job_id": f"JOAW{str(i+1).zfill(4)}",
        "family_id": f"FAM{str((i % 5) + 1).zfill(3)}",
        "sequence": (i % 3) + 1,
        "processing_time": 2.5 + (i % 4),
        "machine_types": [1, 2, 3] if i % 2 == 0 else [2, 3, 4],
        "priority": (i % 3) + 1,
        "is_important": i % 5 == 0,
        "lcd_date": (datetime.now() + timedelta(days=7 + (i % 5))).isoformat(),
        "setup_time": 0.3 + (i % 3) * 0.1
    }
    jobs.append(job)

# Create schedule request
schedule_request = {
    "jobs": jobs,
    "schedule_start": datetime.now().isoformat()
}

print("Creating schedule with 20 jobs...")
print(f"First job: {jobs[0]['job_id']}")
print(f"Last job: {jobs[-1]['job_id']}")

# Send request to API
try:
    response = requests.post(
        "http://localhost:8000/schedule",
        json=schedule_request,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": "dev-api-key-change-in-production"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nSchedule created successfully!")
        print(f"Schedule ID: {result['schedule_id']}")
        print(f"Makespan: {result['metrics']['makespan']:.1f} hours")
        print(f"Jobs scheduled: {result['metrics']['scheduled_jobs']}/{result['metrics']['total_jobs']}")
        print(f"Completion rate: {result['metrics']['completion_rate']:.1f}%")
        print(f"Average utilization: {result['metrics']['average_utilization']:.1f}%")
        
        # Save result for inspection
        with open("test_schedule_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\nFull result saved to test_schedule_result.json")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Connection error: {e}")
    print("Make sure the API server is running on port 8000")