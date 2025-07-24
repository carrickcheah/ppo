"""
Comprehensive test runner for PPO scheduling system
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_data_pipeline import run_data_pipeline_tests
from test_environment_comprehensive import run_environment_tests
from test_ppo_components import run_ppo_component_tests


def run_integration_tests():
    """Run integration tests between components."""
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60)
    
    print("\n=== Testing Data Pipeline → Environment Integration ===")
    
    import json
    from src.data.data_loader import DataLoader
    from src.environment.scheduling_game_env import SchedulingGameEnv
    
    # Create test data
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "jobs": [
            {
                "JoId_v": "TEST001",
                "Task_v": "Cutting",
                "processing_time": 2.5,
                "required_machines": [1],
                "TargetDate_dd": 5,
                "IsImportant": 1
            }
        ],
        "machines": [
            {"MachineId_i": 1, "MachineName_v": "TestMachine"}
        ],
        "working_hours": None
    }
    
    # Save test data
    test_file = "/Users/carrickcheah/Project/ppo/app_2/tests/integration_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    try:
        # Load data
        loader = DataLoader({"source": "snapshot", "snapshot_path": test_file})
        jobs = loader.load_jobs()
        machines = loader.load_machines()
        
        # Create environment
        env = SchedulingGameEnv(jobs, machines, None, {})
        obs, _ = env.reset()
        
        # Test action
        action = [0, 0]  # Schedule job 0 on machine 0
        obs, reward, done, truncated, info = env.step(action)
        
        assert info.get('valid_action', False), "Action should be valid"
        print("✓ Data pipeline → Environment integration successful")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            
    print("\n=== Testing Environment → PPO Model Integration ===")
    
    from phase2.ppo_scheduler import PPOScheduler
    
    # Create simple config
    config = {
        'model': {
            'job_embedding_dim': 32,
            'machine_embedding_dim': 16,
            'hidden_dim': 64,
            'n_heads': 2,
            'n_layers': 1,
            'dropout': 0.1,
            'max_jobs': 10,
            'max_machines': 5
        },
        'ppo': {
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'learning_rate': 3e-4
        },
        'device': 'cpu'
    }
    
    # Create model
    model = PPOScheduler(config)
    
    # Get observation from environment
    obs, _ = env.reset()
    mask = env.get_action_mask()
    
    # Get action from model
    action, value, log_prob = model.get_action(
        obs, mask, n_jobs=len(jobs), n_machines=len(machines)
    )
    
    # Execute action in environment
    obs, reward, done, truncated, info = env.step([action // len(machines), action % len(machines)])
    
    print("✓ Environment → PPO Model integration successful")
    print(f"  - Model selected action: {action}")
    print(f"  - Environment accepted: {info.get('valid_action', False)}")
    print(f"  - Reward received: {reward}")
    
    print("\n✅ All integration tests passed!")


def run_performance_tests():
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTS")
    print("="*60)
    
    from src.environment.scheduling_game_env import SchedulingGameEnv
    from phase2.ppo_scheduler import PPOScheduler
    import numpy as np
    
    # Create larger test case
    n_jobs = 100
    n_machines = 20
    
    jobs = []
    for i in range(n_jobs):
        jobs.append({
            'JoId_v': f'PERF{i:03d}',
            'Task_v': f'Task{i%5}',
            'processing_time': np.random.uniform(1, 10),
            'required_machines': [np.random.randint(1, n_machines+1)],
            'TargetDate_dd': np.random.randint(1, 30),
            'IsImportant': np.random.randint(0, 2)
        })
        
    machines = []
    for i in range(n_machines):
        machines.append({
            'MachineId_i': i+1,
            'MachineName_v': f'M{i+1:02d}'
        })
        
    # Test environment speed
    print(f"\n=== Environment Performance (N={n_jobs} jobs, M={n_machines} machines) ===")
    
    env = SchedulingGameEnv(jobs, machines, None, {})
    
    # Time reset
    start = time.time()
    for _ in range(10):
        env.reset()
    reset_time = (time.time() - start) / 10
    print(f"✓ Reset time: {reset_time*1000:.2f} ms")
    
    # Time steps
    env.reset()
    start = time.time()
    steps = 0
    while steps < 100:
        mask = env.get_action_mask()
        if not np.any(mask):
            break
        valid_idx = np.where(mask)[0][0]
        job_idx = valid_idx // n_machines
        machine_idx = valid_idx % n_machines
        env.step([job_idx, machine_idx])
        steps += 1
    step_time = (time.time() - start) / steps
    print(f"✓ Step time: {step_time*1000:.2f} ms")
    print(f"✓ Steps per second: {1/step_time:.0f}")
    
    # Test model inference speed
    print(f"\n=== Model Inference Performance ===")
    
    config = {
        'model': {
            'job_embedding_dim': 64,
            'machine_embedding_dim': 32,
            'hidden_dim': 128,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.0,
            'max_jobs': 200,
            'max_machines': 50
        },
        'ppo': {
            'clip_range': 0.2,
            'learning_rate': 3e-4
        },
        'device': 'cpu'
    }
    
    model = PPOScheduler(config)
    model.eval()
    
    # Create batch of observations
    batch_size = 16
    obs = np.random.randn(batch_size, n_jobs * 10 + n_machines * 5).astype(np.float32)
    mask = np.ones((batch_size, n_jobs * n_machines), dtype=bool)
    
    # Time inference
    with torch.no_grad():
        start = time.time()
        for i in range(batch_size):
            model.get_action(obs[i], mask[i], n_jobs=n_jobs, n_machines=n_machines)
        inference_time = (time.time() - start) / batch_size
        
    print(f"✓ Inference time: {inference_time*1000:.2f} ms per observation")
    print(f"✓ Inference throughput: {1/inference_time:.0f} obs/sec")
    
    print("\n✅ Performance tests completed!")


def generate_test_report(results):
    """Generate test report."""
    report = []
    report.append("# PPO Scheduling System - Test Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary
    total_tests = sum(len(r['tests']) for r in results)
    passed_tests = sum(sum(1 for t in r['tests'] if t['passed']) for r in results)
    
    report.append("## Summary")
    report.append(f"- Total Tests: {total_tests}")
    report.append(f"- Passed: {passed_tests}")
    report.append(f"- Failed: {total_tests - passed_tests}")
    report.append(f"- Success Rate: {passed_tests/total_tests*100:.1f}%\n")
    
    # Details by category
    report.append("## Test Results by Category\n")
    
    for result in results:
        report.append(f"### {result['category']}")
        report.append(f"- Tests Run: {len(result['tests'])}")
        report.append(f"- Duration: {result['duration']:.2f}s")
        
        if result['errors']:
            report.append("\n**Errors:**")
            for error in result['errors']:
                report.append(f"- {error}")
                
        report.append("")
        
    # Save report
    report_path = "/Users/carrickcheah/Project/ppo/app_2/tests/test_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"\n📄 Test report saved to: {report_path}")


def main():
    """Main test runner."""
    print("\n" + "="*80)
    print("PPO SCHEDULING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test categories
    test_suites = [
        ("Data Pipeline", run_data_pipeline_tests),
        ("Environment", run_environment_tests),
        ("PPO Components", run_ppo_component_tests),
        ("Integration", run_integration_tests),
        ("Performance", run_performance_tests)
    ]
    
    for category, test_func in test_suites:
        print(f"\nRunning {category} tests...")
        start_time = time.time()
        
        result = {
            'category': category,
            'tests': [],
            'errors': [],
            'duration': 0
        }
        
        try:
            test_func()
            result['tests'].append({'name': category, 'passed': True})
        except Exception as e:
            print(f"\n❌ {category} tests failed!")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            result['tests'].append({'name': category, 'passed': False})
            result['errors'].append(str(e))
            
        result['duration'] = time.time() - start_time
        results.append(result)
        
    # Generate report
    generate_test_report(results)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = sum(len(r['tests']) for r in results)
    passed_tests = sum(sum(1 for t in r['tests'] if t['passed']) for r in results)
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✅ ALL TESTS PASSED! The system is ready for training.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Import torch here to avoid issues
    import torch
    main()