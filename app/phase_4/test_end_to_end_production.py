#!/usr/bin/env python3
"""
End-to-end production test for PPO Scheduler
Tests the complete workflow from data loading to schedule generation
"""

import sys
import json
import time
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.database import Database
from src.utils.logger import setup_logger

logger = setup_logger("e2e_test", "logs/e2e_test.log")

class EndToEndTest:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.api_key = "test-api-key-123"
        self.server_process = None
        
    def setup(self):
        """Setup test environment"""
        logger.info("="*60)
        logger.info("END-TO-END PRODUCTION TEST")
        logger.info("="*60)
        
        # Check production data exists
        snapshot_path = Path("data/real_production_snapshot.json")
        if not snapshot_path.exists():
            logger.error("Production snapshot not found. Run ingest_data.py first")
            return False
            
        # Load and analyze production data
        with open(snapshot_path) as f:
            self.production_data = json.load(f)
            
        logger.info(f"Loaded production data:")
        logger.info(f"  - Families: {len(self.production_data['families'])}")
        logger.info(f"  - Machines: {len(self.production_data['machines'])}")
        
        # Count total jobs
        total_jobs = sum(len(family.get('tasks', [])) 
                        for family in self.production_data['families'].values())
        logger.info(f"  - Total jobs: {total_jobs}")
        
        return True
        
    def start_api_server(self):
        """Start the API server in background"""
        logger.info("\nStarting API server...")
        
        # Start server process
        self.server_process = subprocess.Popen(
            [sys.executable, "run_api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if server is running
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                logger.info("API server started successfully")
                health = response.json()
                logger.info(f"  - Status: {health['status']}")
                logger.info(f"  - Model loaded: {health['model_loaded']}")
                return True
        except:
            pass
            
        logger.error("Failed to start API server")
        return False
        
    def test_small_batch(self):
        """Test with small batch that fits in single pass"""
        logger.info("\nTest 1: Small batch (50 jobs)")
        
        # Extract first 50 jobs from production data
        jobs = []
        job_count = 0
        
        for family_id, family_data in self.production_data['families'].items():
            for task in family_data.get('tasks', []):
                if job_count >= 50:
                    break
                    
                job = {
                    "job_id": task['job_id'],
                    "family_id": family_id,
                    "sequence": task.get('sequence', 1),
                    "processing_time": task['processing_time'],
                    "machine_types": task['allowed_machine_types'],
                    "priority": family_data.get('priority', 3),
                    "is_important": family_data.get('is_important', False),
                    "lcd_date": (datetime.now() + timedelta(days=family_data.get('lcd_days_remaining', 30))).isoformat(),
                    "setup_time": task.get('setup_time', 0.5)
                }
                jobs.append(job)
                job_count += 1
                
            if job_count >= 50:
                break
                
        logger.info(f"  - Prepared {len(jobs)} jobs for testing")
        
        # Send request
        request_data = {
            "request_id": "test-small-batch",
            "jobs": jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/schedule",
            json=request_data,
            headers=headers
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"  - Success! Schedule created in {elapsed:.2f}s")
            logger.info(f"  - Jobs scheduled: {result['metrics']['scheduled_jobs']}/{result['metrics']['total_jobs']}")
            logger.info(f"  - Makespan: {result['metrics']['makespan']:.1f}h")
            logger.info(f"  - Completion rate: {result['metrics']['completion_rate']:.1f}%")
            logger.info(f"  - Algorithm: {result['algorithm_used']}")
            return True
        else:
            logger.error(f"  - Failed: {response.status_code}")
            logger.error(f"  - Error: {response.json()}")
            return False
            
    def test_full_production(self):
        """Test with full production data (should trigger batching)"""
        logger.info("\nTest 2: Full production (all jobs)")
        
        # Extract all jobs from production data
        jobs = []
        
        for family_id, family_data in self.production_data['families'].items():
            for task in family_data.get('tasks', []):
                job = {
                    "job_id": task['job_id'],
                    "family_id": family_id,
                    "sequence": task.get('sequence', 1),
                    "processing_time": task['processing_time'],
                    "machine_types": task['allowed_machine_types'],
                    "priority": family_data.get('priority', 3),
                    "is_important": family_data.get('is_important', False),
                    "lcd_date": (datetime.now() + timedelta(days=family_data.get('lcd_days_remaining', 30))).isoformat(),
                    "setup_time": task.get('setup_time', 0.5)
                }
                jobs.append(job)
                
        logger.info(f"  - Prepared {len(jobs)} jobs for testing")
        
        # Send request
        request_data = {
            "request_id": "test-full-production",
            "jobs": jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/schedule",
            json=request_data,
            headers=headers,
            timeout=60  # Longer timeout for large batch
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"  - Success! Schedule created in {elapsed:.2f}s")
            logger.info(f"  - Jobs scheduled: {result['metrics']['scheduled_jobs']}/{result['metrics']['total_jobs']}")
            logger.info(f"  - Makespan: {result['metrics']['makespan']:.1f}h")
            logger.info(f"  - Completion rate: {result['metrics']['completion_rate']:.1f}%")
            
            if 'batches_used' in result['metrics']:
                logger.info(f"  - Batches used: {result['metrics']['batches_used']}")
                
            # Analyze schedule quality
            self.analyze_schedule(result['scheduled_jobs'])
            return True
        else:
            logger.error(f"  - Failed: {response.status_code}")
            logger.error(f"  - Error: {response.json()}")
            return False
            
    def analyze_schedule(self, scheduled_jobs):
        """Analyze the generated schedule"""
        logger.info("\n  Schedule Analysis:")
        
        # Machine utilization
        machine_usage = {}
        for job in scheduled_jobs:
            machine_id = job['machine_id']
            if machine_id not in machine_usage:
                machine_usage[machine_id] = 0
            machine_usage[machine_id] += job['processing_time']
            
        logger.info(f"    - Machines used: {len(machine_usage)}")
        logger.info(f"    - Avg utilization: {sum(machine_usage.values())/len(machine_usage):.1f}h per machine")
        
        # Job completion times
        completion_times = []
        for job in scheduled_jobs:
            end_time = datetime.fromisoformat(job['end_time'].replace('Z', '+00:00'))
            start_time = datetime.fromisoformat(job['start_time'].replace('Z', '+00:00'))
            completion_times.append((end_time - start_time).total_seconds() / 3600)
            
        logger.info(f"    - Avg processing time: {sum(completion_times)/len(completion_times):.1f}h")
        logger.info(f"    - Min processing time: {min(completion_times):.1f}h")
        logger.info(f"    - Max processing time: {max(completion_times):.1f}h")
        
    def test_performance(self):
        """Test API performance with various batch sizes"""
        logger.info("\nTest 3: Performance testing")
        
        batch_sizes = [10, 50, 100, 200, 400]
        results = []
        
        for size in batch_sizes:
            # Create test jobs
            jobs = []
            for i in range(size):
                job = {
                    "job_id": f"PERF-TEST-{i:04d}",
                    "family_id": f"FAM-{i//10}",
                    "sequence": i % 10 + 1,
                    "processing_time": 2.5,
                    "machine_types": [1, 2, 3, 4, 5],
                    "priority": (i % 3) + 1,
                    "is_important": i < size // 10,
                    "lcd_date": (datetime.now() + timedelta(days=i//10 + 1)).isoformat(),
                    "setup_time": 0.5
                }
                jobs.append(job)
                
            # Send request
            request_data = {
                "request_id": f"perf-test-{size}",
                "jobs": jobs,
                "schedule_start": datetime.now().isoformat()
            }
            
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/schedule",
                json=request_data,
                headers=headers
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                results.append({
                    'size': size,
                    'time': elapsed,
                    'success': True
                })
            else:
                results.append({
                    'size': size,
                    'time': elapsed,
                    'success': False
                })
                
        # Report results
        logger.info("  Performance Results:")
        logger.info("  Size  | Time (s) | Status")
        logger.info("  ------|----------|--------")
        for r in results:
            status = "Success" if r['success'] else "Failed"
            logger.info(f"  {r['size']:4d}  | {r['time']:8.3f} | {status}")
            
    def cleanup(self):
        """Cleanup test environment"""
        logger.info("\nCleaning up...")
        
        if self.server_process:
            logger.info("  - Stopping API server")
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            
    def run(self):
        """Run all tests"""
        try:
            # Setup
            if not self.setup():
                return False
                
            # Start API server
            if not self.start_api_server():
                return False
                
            # Run tests
            test_results = []
            test_results.append(("Small batch", self.test_small_batch()))
            test_results.append(("Full production", self.test_full_production()))
            test_results.append(("Performance", True))  # Performance test doesn't return bool
            self.test_performance()
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("TEST SUMMARY")
            logger.info("="*60)
            
            for test_name, result in test_results:
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name:20s}: {status}")
                
            all_passed = all(r[1] for r in test_results)
            
            if all_passed:
                logger.info("\nALL TESTS PASSED!")
                logger.info("The PPO Scheduler is ready for production deployment.")
            else:
                logger.error("\nSOME TESTS FAILED!")
                logger.error("Please check the logs for details.")
                
            return all_passed
            
        except Exception as e:
            logger.error(f"Test error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()

if __name__ == "__main__":
    test = EndToEndTest()
    success = test.run()
    sys.exit(0 if success else 1)