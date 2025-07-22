#!/usr/bin/env python3
"""
Test the API integration with PPO scheduler
"""

import requests
import json
from datetime import datetime, timedelta

# API configuration
API_URL = "http://localhost:8000"
API_KEY = "test-api-key-123"

print("="*60)
print("TESTING API WITH PPO SCHEDULER")
print("="*60)

# Test 1: Health check
print("\n1. Testing health endpoint...")
response = requests.get(f"{API_URL}/health")
if response.status_code == 200:
    health = response.json()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Database connected: {health['database_connected']}")
else:
    print(f"   Error: {response.status_code}")

# Test 2: Schedule with few jobs (should work in single batch)
print("\n2. Testing schedule with 100 jobs...")
jobs = []
for i in range(100):
    jobs.append({
        "job_id": f"TEST-{i:04d}",
        "family_id": f"FAM-{i//10}",
        "sequence": i % 10 + 1,
        "processing_time": 2.5,
        "machine_types": [1, 2, 3],
        "priority": (i % 3) + 1,
        "is_important": i < 30,
        "lcd_date": (datetime.now() + timedelta(days=i//10 + 1)).isoformat(),
        "setup_time": 0.5
    })

request_data = {
    "request_id": "test-100-jobs",
    "jobs": jobs,
    "schedule_start": datetime.now().isoformat()
}

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(
    f"{API_URL}/schedule",
    json=request_data,
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"   Schedule ID: {result['schedule_id']}")
    print(f"   Jobs scheduled: {result['metrics']['scheduled_jobs']}/{result['metrics']['total_jobs']}")
    print(f"   Makespan: {result['metrics']['makespan']:.1f}h")
    print(f"   Completion rate: {result['metrics']['completion_rate']:.1f}%")
    print(f"   Algorithm: {result['algorithm_used']}")
else:
    print(f"   Error: {response.status_code}")
    print(f"   Detail: {response.json()}")

# Test 3: Schedule with many jobs (should trigger batch processing)
print("\n3. Testing schedule with 400 jobs...")
jobs = []
for i in range(400):
    jobs.append({
        "job_id": f"TEST-{i:04d}",
        "family_id": f"FAM-{i//10}",
        "sequence": i % 10 + 1,
        "processing_time": 2.5,
        "machine_types": [1, 2, 3, 4, 5],
        "priority": (i % 4) + 1,
        "is_important": i < 100,
        "lcd_date": (datetime.now() + timedelta(days=i//20 + 1)).isoformat(),
        "setup_time": 0.3
    })

request_data = {
    "request_id": "test-400-jobs",
    "jobs": jobs,
    "schedule_start": datetime.now().isoformat()
}

response = requests.post(
    f"{API_URL}/schedule",
    json=request_data,
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"   Schedule ID: {result['schedule_id']}")
    print(f"   Jobs scheduled: {result['metrics']['scheduled_jobs']}/{result['metrics']['total_jobs']}")
    print(f"   Makespan: {result['metrics']['makespan']:.1f}h")
    print(f"   Completion rate: {result['metrics']['completion_rate']:.1f}%")
    
    # Check if batch processing was used
    if 'batches_used' in result['metrics']:
        print(f"   Batches used: {result['metrics']['batches_used']}")
    
    if result.get('warnings'):
        print(f"   Warnings: {result['warnings']}")
else:
    print(f"   Error: {response.status_code}")
    print(f"   Detail: {response.json()}")

print("\n" + "="*60)
print("API INTEGRATION TEST COMPLETE")
print("="*60)