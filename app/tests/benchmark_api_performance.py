#!/usr/bin/env python3
"""
Performance Benchmarking for PPO Scheduler API

Measures API performance metrics including:
- Response times
- Throughput
- Resource usage
- Scheduling quality
"""

import time
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import requests
import concurrent.futures
from dataclasses import dataclass, asdict

# API Configuration
API_URL = "http://localhost:8000"
API_KEY = "dev-api-key-change-in-production"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    job_count: int
    response_time: float
    makespan: float
    completion_rate: float
    scheduled_jobs: int
    utilization: float
    timestamp: str
    error: str = None


class APIBenchmark:
    """Performance benchmarking for the PPO Scheduler API"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.ensure_api_running()
    
    def ensure_api_running(self):
        """Check if API is accessible"""
        try:
            response = requests.get(f"{API_URL}/health", headers=HEADERS, timeout=5)
            if response.status_code != 200:
                raise Exception("API not healthy")
            data = response.json()
            if not data.get("model_loaded"):
                raise Exception("Model not loaded")
            print(f"API Status: {data['status']}")
            print(f"Model Loaded: {data['model_loaded']}")
            print(f"Environment: {data['environment']}")
            print("-" * 50)
        except Exception as e:
            print(f"ERROR: API not accessible - {e}")
            print("Start the API with: uv run python run_api_server.py")
            exit(1)
    
    def benchmark_single_request(self, job_count: int) -> BenchmarkResult:
        """Benchmark a single scheduling request"""
        jobs = self.generate_jobs(job_count)
        
        request_data = {
            "jobs": jobs,
            "schedule_start": datetime.now().isoformat()
        }
        
        # Measure response time
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/schedule",
                headers=HEADERS,
                json=request_data,
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                metrics = data["metrics"]
                
                return BenchmarkResult(
                    job_count=job_count,
                    response_time=response_time,
                    makespan=metrics.get("makespan", 0),
                    completion_rate=metrics.get("completion_rate", 0),
                    scheduled_jobs=metrics.get("scheduled_jobs", 0),
                    utilization=metrics.get("average_utilization", 0),
                    timestamp=datetime.now().isoformat()
                )
            else:
                return BenchmarkResult(
                    job_count=job_count,
                    response_time=response_time,
                    makespan=0,
                    completion_rate=0,
                    scheduled_jobs=0,
                    utilization=0,
                    timestamp=datetime.now().isoformat(),
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return BenchmarkResult(
                job_count=job_count,
                response_time=time.time() - start_time,
                makespan=0,
                completion_rate=0,
                scheduled_jobs=0,
                utilization=0,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    def benchmark_job_scaling(self):
        """Test performance with increasing job counts"""
        print("Running Job Scaling Benchmark...")
        job_counts = [10, 25, 50, 100, 150, 170]  # Up to environment limit
        
        for count in job_counts:
            print(f"\nTesting with {count} jobs...")
            
            # Run 3 iterations for each job count
            iteration_results = []
            for i in range(3):
                result = self.benchmark_single_request(count)
                iteration_results.append(result)
                self.results.append(result)
                
                if not result.error:
                    print(f"  Iteration {i+1}: {result.response_time:.2f}s, "
                          f"Makespan: {result.makespan:.1f}h, "
                          f"Completion: {result.completion_rate:.1f}%")
                else:
                    print(f"  Iteration {i+1}: ERROR - {result.error}")
                
                time.sleep(1)  # Brief pause between iterations
            
            # Calculate averages
            valid_results = [r for r in iteration_results if not r.error]
            if valid_results:
                avg_response = statistics.mean(r.response_time for r in valid_results)
                avg_makespan = statistics.mean(r.makespan for r in valid_results)
                avg_completion = statistics.mean(r.completion_rate for r in valid_results)
                
                print(f"\n  Averages for {count} jobs:")
                print(f"    Response Time: {avg_response:.2f}s")
                print(f"    Makespan: {avg_makespan:.1f}h")
                print(f"    Completion Rate: {avg_completion:.1f}%")
    
    def benchmark_concurrent_load(self):
        """Test API under concurrent load"""
        print("\n" + "="*50)
        print("Running Concurrent Load Benchmark...")
        
        concurrent_levels = [1, 2, 5, 10]
        job_count = 50  # Fixed job count for consistency
        
        for level in concurrent_levels:
            print(f"\nTesting with {level} concurrent requests...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                start_time = time.time()
                futures = [
                    executor.submit(self.benchmark_single_request, job_count)
                    for _ in range(level)
                ]
                results = [f.result() for f in futures]
                total_time = time.time() - start_time
            
            # Analyze results
            successful = [r for r in results if not r.error]
            failed = len(results) - len(successful)
            
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Successful: {len(successful)}/{level}")
            
            if successful:
                avg_response = statistics.mean(r.response_time for r in successful)
                print(f"  Avg Response Time: {avg_response:.2f}s")
                print(f"  Throughput: {len(successful)/total_time:.2f} req/s")
            
            if failed > 0:
                print(f"  Failed Requests: {failed}")
            
            self.results.extend(results)
            time.sleep(2)  # Recovery time between tests
    
    def benchmark_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n" + "="*50)
        print("Running Edge Case Benchmarks...")
        
        # Test 1: Empty job list
        print("\nTest 1: Empty job list")
        result = self.benchmark_single_request(0)
        print(f"  Response Time: {result.response_time:.2f}s")
        print(f"  Error: {result.error or 'None'}")
        
        # Test 2: Single job
        print("\nTest 2: Single job")
        result = self.benchmark_single_request(1)
        print(f"  Response Time: {result.response_time:.2f}s")
        if not result.error:
            print(f"  Completion Rate: {result.completion_rate:.1f}%")
        
        # Test 3: Over limit (should handle gracefully)
        print("\nTest 3: Over environment limit (200 jobs)")
        result = self.benchmark_single_request(200)
        print(f"  Response Time: {result.response_time:.2f}s")
        if not result.error:
            print(f"  Scheduled: {result.scheduled_jobs}/{200}")
    
    def analyze_results(self):
        """Analyze and summarize benchmark results"""
        print("\n" + "="*50)
        print("Benchmark Analysis")
        print("="*50)
        
        # Group by job count
        job_groups = {}
        for result in self.results:
            if result.job_count not in job_groups:
                job_groups[result.job_count] = []
            job_groups[result.job_count].append(result)
        
        print("\nPerformance by Job Count:")
        print(f"{'Jobs':<8} {'Avg Response':<15} {'Avg Makespan':<15} {'Avg Completion':<15}")
        print("-" * 53)
        
        for job_count in sorted(job_groups.keys()):
            results = job_groups[job_count]
            valid = [r for r in results if not r.error]
            
            if valid:
                avg_response = statistics.mean(r.response_time for r in valid)
                avg_makespan = statistics.mean(r.makespan for r in valid)
                avg_completion = statistics.mean(r.completion_rate for r in valid)
                
                print(f"{job_count:<8} {avg_response:<15.2f} {avg_makespan:<15.1f} {avg_completion:<15.1f}")
        
        # Overall statistics
        all_valid = [r for r in self.results if not r.error]
        if all_valid:
            print(f"\nOverall Statistics ({len(all_valid)} successful requests):")
            print(f"  Response Time: min={min(r.response_time for r in all_valid):.2f}s, "
                  f"max={max(r.response_time for r in all_valid):.2f}s, "
                  f"avg={statistics.mean(r.response_time for r in all_valid):.2f}s")
            print(f"  Completion Rate: min={min(r.completion_rate for r in all_valid):.1f}%, "
                  f"max={max(r.completion_rate for r in all_valid):.1f}%, "
                  f"avg={statistics.mean(r.completion_rate for r in all_valid):.1f}%")
        
        # Error analysis
        errors = [r for r in self.results if r.error]
        if errors:
            print(f"\nErrors: {len(errors)}/{len(self.results)} requests failed")
            error_types = {}
            for e in errors:
                error_type = e.error.split()[0] if e.error else "Unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count}")
    
    def save_results(self):
        """Save benchmark results to file"""
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"tests/{filename}"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "api_url": API_URL,
            "total_requests": len(self.results),
            "successful_requests": len([r for r in self.results if not r.error]),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def generate_jobs(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic job data"""
        jobs = []
        prefixes = ["JOAW", "JOST", "JOEX"]
        
        for i in range(count):
            prefix = prefixes[i % len(prefixes)]
            family_num = i // 5
            
            # Vary job characteristics
            processing_time = 1.0 + (i % 10) * 0.5
            if i % 7 == 0:  # Some long jobs
                processing_time *= 2
            
            machine_types = self._get_machine_types(i)
            
            job = {
                "job_id": f"{prefix}{i:04d}",
                "family_id": f"FAM{family_num:03d}",
                "sequence": (i % 5) + 1,
                "processing_time": processing_time,
                "machine_types": machine_types,
                "priority": (i % 3) + 1,
                "is_important": i % 4 == 0,
                "lcd_date": (datetime.now() + timedelta(days=1 + i % 7)).isoformat(),
                "setup_time": 0.3 if i % 2 == 0 else 0.5
            }
            jobs.append(job)
        
        return jobs
    
    def _get_machine_types(self, index: int) -> List[int]:
        """Get varied machine type combinations"""
        patterns = [
            [1, 2, 3],      # Common machines
            [2, 3, 4],      # Alternative set
            [1, 3],         # Limited options
            [2, 4],         # Different limited set
            [1, 2, 3, 4],   # All machines
            [3, 4],         # Specialized machines
            [1, 2],         # Basic machines
            [2, 3],         # Mid-range machines
            [1],            # Single machine option
            [4]             # Specialized only
        ]
        return patterns[index % len(patterns)]
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("PPO Scheduler API Performance Benchmark")
        print("="*50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        # Run all benchmarks
        self.benchmark_job_scaling()
        self.benchmark_concurrent_load()
        self.benchmark_edge_cases()
        
        # Analyze and save
        self.analyze_results()
        self.save_results()
        
        print("\nBenchmark complete!")


if __name__ == "__main__":
    benchmark = APIBenchmark()
    benchmark.run_full_benchmark()