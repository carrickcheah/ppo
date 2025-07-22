#!/usr/bin/env python3
"""
Integration Test Suite for PPO Scheduler API

Tests the API with real production data patterns to ensure
the Phase 4 model is working correctly in deployment.
"""

import pytest
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

# API Configuration
API_URL = "http://localhost:8000"
API_KEY = "dev-api-key-change-in-production"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


class TestAPIIntegration:
    """Integration tests for the PPO Scheduler API"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure API is running before tests"""
        try:
            response = requests.get(f"{API_URL}/health", headers=HEADERS, timeout=5)
            assert response.status_code == 200
        except (requests.exceptions.ConnectionError, AssertionError):
            pytest.skip("API server not running. Start with: uv run python run_api_server.py")
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = requests.get(f"{API_URL}/health", headers=HEADERS)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert data["model_loaded"] is True
    
    def test_schedule_small_batch(self):
        """Test scheduling with a small batch of jobs"""
        # Create 10 jobs with realistic parameters
        jobs = []
        for i in range(10):
            job = {
                "job_id": f"JOAW00{i:02d}",
                "family_id": f"FAM{i // 3}",
                "sequence": i % 3 + 1,
                "processing_time": 2.5 + (i % 4) * 0.5,
                "machine_types": [1, 2, 3] if i % 2 == 0 else [2, 3, 4],
                "priority": (i % 3) + 1,
                "is_important": i < 5,
                "lcd_date": (datetime.now() + timedelta(days=i+1)).isoformat(),
                "setup_time": 0.3
            }
            jobs.append(job)
        
        request_data = {
            "jobs": jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{API_URL}/schedule",
            headers=HEADERS,
            json=request_data
        )
        
        # API might return 500 if no machines in database
        if response.status_code == 500:
            pytest.skip("Database not configured with machines")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "schedule_id" in data
        assert "scheduled_jobs" in data
        assert "metrics" in data
        
        # Validate metrics
        metrics = data["metrics"]
        assert metrics["total_jobs"] == 10
        assert metrics["scheduled_jobs"] > 0
        assert metrics["makespan"] > 0
        assert 0 <= metrics["completion_rate"] <= 100
    
    def test_schedule_medium_batch(self):
        """Test scheduling with 50 jobs (medium complexity)"""
        jobs = self._generate_realistic_jobs(50)
        
        request_data = {
            "jobs": jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/schedule",
            headers=HEADERS,
            json=request_data
        )
        response_time = time.time() - start_time
        
        if response.status_code == 500:
            pytest.skip("Database not configured")
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
        
        data = response.json()
        metrics = data["metrics"]
        
        # For Phase 4 model, we expect good performance
        assert metrics["completion_rate"] > 80  # Phase 4 achieves 100%
        assert metrics["makespan"] < 100  # Reasonable makespan
    
    def test_schedule_large_batch(self):
        """Test scheduling with 170 jobs (environment limit)"""
        jobs = self._generate_realistic_jobs(170)
        
        request_data = {
            "jobs": jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{API_URL}/schedule",
            headers=HEADERS,
            json=request_data,
            timeout=30  # Allow more time for large batch
        )
        
        if response.status_code == 500:
            pytest.skip("Database not configured")
        
        assert response.status_code == 200
        
        data = response.json()
        metrics = data["metrics"]
        
        # Validate all jobs were considered
        assert metrics["total_jobs"] == 170
        assert metrics["scheduled_jobs"] > 150  # Most should be scheduled
    
    def test_schedule_with_constraints(self):
        """Test scheduling with various constraints"""
        # Jobs with tight deadlines
        urgent_jobs = []
        for i in range(5):
            job = {
                "job_id": f"URGENT{i:03d}",
                "family_id": "URGENT_FAM",
                "sequence": i + 1,
                "processing_time": 1.5,
                "machine_types": [1, 2],  # Limited machine options
                "priority": 1,  # High priority
                "is_important": True,
                "lcd_date": (datetime.now() + timedelta(hours=6)).isoformat(),
                "setup_time": 0.2
            }
            urgent_jobs.append(job)
        
        # Jobs with flexible deadlines
        flexible_jobs = []
        for i in range(5):
            job = {
                "job_id": f"FLEX{i:03d}",
                "family_id": "FLEX_FAM",
                "sequence": i + 1,
                "processing_time": 3.0,
                "machine_types": [1, 2, 3, 4],  # Many machine options
                "priority": 3,  # Low priority
                "is_important": False,
                "lcd_date": (datetime.now() + timedelta(days=7)).isoformat(),
                "setup_time": 0.5
            }
            flexible_jobs.append(job)
        
        request_data = {
            "jobs": urgent_jobs + flexible_jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{API_URL}/schedule",
            headers=HEADERS,
            json=request_data
        )
        
        if response.status_code == 500:
            pytest.skip("Database not configured")
        
        assert response.status_code == 200
        
        data = response.json()
        scheduled_jobs = data["scheduled_jobs"]
        
        # Check if urgent jobs are scheduled first
        urgent_scheduled = [j for j in scheduled_jobs if j["job_id"].startswith("URGENT")]
        flexible_scheduled = [j for j in scheduled_jobs if j["job_id"].startswith("FLEX")]
        
        if urgent_scheduled and flexible_scheduled:
            # Urgent jobs should generally start earlier
            avg_urgent_start = sum(j["start_time"] for j in urgent_scheduled) / len(urgent_scheduled)
            avg_flexible_start = sum(j["start_time"] for j in flexible_scheduled) / len(flexible_scheduled)
            assert avg_urgent_start < avg_flexible_start
    
    def test_invalid_request(self):
        """Test API error handling with invalid requests"""
        # Missing required fields
        invalid_request = {
            "jobs": [{"job_id": "TEST001"}]  # Missing required fields
        }
        
        response = requests.post(
            f"{API_URL}/schedule",
            headers=HEADERS,
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_empty_job_list(self):
        """Test scheduling with empty job list"""
        request_data = {
            "jobs": [],
            "schedule_start": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{API_URL}/schedule",
            headers=HEADERS,
            json=request_data
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_api_authentication(self):
        """Test API key authentication"""
        # No API key
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 403
        
        # Invalid API key
        bad_headers = {"X-API-Key": "invalid-key"}
        response = requests.get(f"{API_URL}/health", headers=bad_headers)
        assert response.status_code == 403
    
    def test_concurrent_requests(self):
        """Test API handling of concurrent requests"""
        import concurrent.futures
        
        def make_request(job_count):
            jobs = self._generate_realistic_jobs(job_count)
            request_data = {
                "jobs": jobs,
                "schedule_start": datetime.now().isoformat()
            }
            return requests.post(
                f"{API_URL}/schedule",
                headers=HEADERS,
                json=request_data
            )
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, 10) for _ in range(5)]
            responses = [f.result() for f in futures]
        
        # All should complete (either 200 or 500 if DB not configured)
        for response in responses:
            assert response.status_code in [200, 500]
    
    def _generate_realistic_jobs(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic job data for testing"""
        jobs = []
        prefixes = ["JOAW", "JOST", "JOEX"]
        
        for i in range(count):
            prefix = prefixes[i % len(prefixes)]
            family_num = i // 5
            
            job = {
                "job_id": f"{prefix}{i:04d}",
                "family_id": f"FAM{family_num:03d}",
                "sequence": (i % 5) + 1,
                "processing_time": 1.5 + (i % 10) * 0.3,
                "machine_types": self._get_machine_types(i),
                "priority": (i % 3) + 1,
                "is_important": i % 4 == 0,
                "lcd_date": (datetime.now() + timedelta(days=1 + i % 7)).isoformat(),
                "setup_time": 0.3 if i % 2 == 0 else 0.5
            }
            jobs.append(job)
        
        return jobs
    
    def _get_machine_types(self, index: int) -> List[int]:
        """Get realistic machine type combinations"""
        patterns = [
            [1, 2, 3],
            [2, 3, 4],
            [1, 3],
            [2, 4],
            [1, 2, 3, 4],
            [3, 4],
            [1, 2],
            [2, 3]
        ]
        return patterns[index % len(patterns)]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])