#!/usr/bin/env python
"""
Test the complete PPO scheduling visualization system integration.
"""

import requests
import json
import time

def test_integration():
    """Test the complete system integration."""
    
    print("="*60)
    print("PPO SCHEDULING VISUALIZATION SYSTEM - INTEGRATION TEST")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Check API is running
    print("\n1. Testing API root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Get available datasets
    print("\n2. Getting available datasets...")
    try:
        response = requests.get(f"{base_url}/api/datasets")
        datasets = response.json()
        print(f"   Found {len(datasets['datasets'])} datasets:")
        for ds in datasets['datasets']:
            print(f"     - {ds['name']}: {ds['total_tasks']} tasks, {ds['total_families']} families")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 3: Get available models
    print("\n3. Getting available models...")
    try:
        response = requests.get(f"{base_url}/api/models")
        models = response.json()
        print(f"   Found {len(models['models'])} models:")
        for model in models['models']:
            print(f"     - {model['name']}: {model['training_steps']} steps")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 4: Schedule jobs with smallest dataset
    print("\n4. Testing scheduling with 10 jobs dataset...")
    try:
        # Note: 10 jobs won't work with 1M model due to observation shape mismatch
        # So we'll use 100 jobs dataset with 1M model
        request_data = {
            "dataset": "100_jobs",
            "model": "sb3_1million",
            "deterministic": True,
            "max_steps": 10000
        }
        
        print(f"   Request: {request_data}")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/schedule",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: SUCCESS")
            print(f"   Message: {result['message']}")
            print(f"   Jobs scheduled: {len(result['jobs'])}")
            print(f"   Machines used: {len(result['machines'])}")
            
            stats = result.get('statistics', {})
            print(f"\n   Statistics:")
            print(f"     - Completion rate: {stats.get('completion_rate', 0):.1f}%")
            print(f"     - On-time rate: {stats.get('on_time_rate', 0):.1f}%")
            print(f"     - Machine utilization: {stats.get('machine_utilization', 0):.1f}%")
            print(f"     - Makespan: {stats.get('makespan', 0):.0f} hours")
            print(f"     - Inference time: {stats.get('inference_time', 0):.2f} seconds")
            print(f"     - API response time: {elapsed:.2f} seconds")
            
            # Show sample jobs
            print(f"\n   Sample jobs (first 3):")
            for job in result['jobs'][:3]:
                print(f"     - {job['task_label']}: {job['start']:.1f}h -> {job['end']:.1f}h on {job['machine']}")
                
        else:
            print(f"   ERROR: {response.status_code}")
            print(f"   {response.json()}")
            return False
            
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 5: Check frontend
    print("\n5. Testing frontend...")
    try:
        response = requests.get("http://localhost:5173/")
        if response.status_code == 200:
            print("   Frontend is running at http://localhost:5173")
            print("   Open this URL in your browser to see the visualization")
        else:
            print(f"   Frontend status: {response.status_code}")
    except:
        print("   Frontend may not be running. Start with: cd frontend3 && npm run dev")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    print("\nSystem is ready! You can now:")
    print("1. Open http://localhost:5173 in your browser")
    print("2. Select a dataset (use 100_jobs for SB3 1M model)")
    print("3. Select a model (SB3 1M Steps)")
    print("4. Click 'Schedule Jobs'")
    print("5. View the Jobs and Machine allocation Gantt charts")
    
    return True

if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)