"""
Comprehensive Test Suite for PPO Scheduling System
Tests all components and generates detailed report
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Component imports
from src.data.db_connector import DBConnector
from src.data.data_loader import DataLoader
from src.environment.scheduling_game_env import SchedulingGameEnv
from src.environment.rules_engine import RulesEngine
from src.environment.reward_function import RewardFunction
from phase2.state_encoder import StateEncoder
from phase2.transformer_policy import TransformerSchedulingPolicy
from phase2.action_masking import ActionMasking
from phase2.ppo_scheduler import PPOScheduler
from phase2.rollout_buffer import RolloutBuffer
from phase2.curriculum import CurriculumManager


class ComponentTest:
    """Base class for component testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.tests = []
        self.results = []
        self.start_time = None
        self.end_time = None
        
    def add_test(self, test_name: str, test_func):
        """Add a test to run"""
        self.tests.append((test_name, test_func))
        
    def run(self) -> Dict[str, Any]:
        """Run all tests for this component"""
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Testing {self.name}")
        print(f"{'='*60}")
        
        passed = 0
        failed = 0
        
        for test_name, test_func in self.tests:
            try:
                print(f"\n• {test_name}...", end=' ')
                test_func()
                print("✓ PASSED")
                self.results.append({
                    'test': test_name,
                    'status': 'PASSED',
                    'error': None
                })
                passed += 1
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                self.results.append({
                    'test': test_name,
                    'status': 'FAILED',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                failed += 1
                
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        return {
            'component': self.name,
            'total_tests': len(self.tests),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(self.tests) * 100 if self.tests else 0,
            'duration': duration,
            'results': self.results
        }


def test_data_layer():
    """Test data layer components"""
    test = ComponentTest("Data Layer")
    
    # Test 1: DataLoader with test data
    def test_data_loader():
        # Create test snapshot
        test_data = {
            "jobs": [{
                "job_id": "TEST001",
                "family_id": "FAM1", 
                "sequence": 1,
                "required_machines": [1],
                "processing_time": 2.0,
                "lcd_days_remaining": 5,
                "is_important": True
            }],
            "machines": [{
                "machine_id": 1,
                "machine_name": "M1",
                "machine_type_id": 1
            }],
            "working_hours": None
        }
        
        # Save and load
        with open("test_snapshot.json", "w") as f:
            json.dump(test_data, f)
            
        loader = DataLoader({"source": "snapshot", "snapshot_path": "test_snapshot.json"})
        jobs = loader.load_jobs()
        machines = loader.load_machines()
        
        assert len(jobs) == 1, f"Expected 1 job, got {len(jobs)}"
        assert len(machines) == 1, f"Expected 1 machine, got {len(machines)}"
        assert jobs[0]['job_id'] == "TEST001"
        
        # Cleanup
        os.remove("test_snapshot.json")
    
    test.add_test("DataLoader basic functionality", test_data_loader)
    
    # Test 2: Production data loading
    def test_production_data():
        snapshot_path = "data/real_production_snapshot.json"
        if os.path.exists(snapshot_path):
            loader = DataLoader({"source": "snapshot", "snapshot_path": snapshot_path})
            jobs = loader.load_jobs()
            machines = loader.load_machines()
            
            assert len(jobs) > 0, "No jobs loaded from production snapshot"
            assert len(machines) > 0, "No machines loaded from production snapshot"
            # Note: May get 0 jobs due to data structure
        else:
            raise FileNotFoundError("Production snapshot not found")
    
    test.add_test("Production data loading", test_production_data)
    
    return test.run()


def test_environment():
    """Test environment components"""
    test = ComponentTest("Environment")
    
    # Test 1: Basic environment creation
    def test_env_creation():
        jobs = [{
            "job_id": "TEST001",
            "family_id": "FAM1",
            "sequence": 1,
            "required_machines": [1],
            "processing_time": 2.0,
            "lcd_days_remaining": 5,
            "is_important": True
        }]
        machines = [{
            "machine_id": 1,
            "machine_name": "M1",
            "machine_type_id": 1
        }]
        
        env = SchedulingGameEnv(jobs, machines, None, {})
        assert env.n_jobs == 1
        assert env.n_machines == 1
    
    test.add_test("Environment creation", test_env_creation)
    
    # Test 2: Environment reset
    def test_env_reset():
        jobs = [{
            "job_id": "TEST001",
            "family_id": "FAM1",
            "sequence": 1,
            "required_machines": [1],
            "processing_time": 2.0,
            "lcd_days_remaining": 5,
            "is_important": True
        }]
        machines = [{
            "machine_id": 1,
            "machine_name": "M1", 
            "machine_type_id": 1
        }]
        
        env = SchedulingGameEnv(jobs, machines, None, {})
        obs, info = env.reset()
        
        assert obs.shape[0] > 0, "Invalid observation shape"
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
    
    test.add_test("Environment reset", test_env_reset)
    
    # Test 3: Rules engine
    def test_rules_engine():
        config = {
            'enforce_sequence': True,
            'enforce_compatibility': True,
            'enforce_no_overlap': True
        }
        rules = RulesEngine(config)
        
        # Test valid action
        job = {'job_id': 'J1', 'sequence': 1, 'family_id': 'FAM1', 'required_machines': [1]}
        machine = {'machine_id': 1}
        current_schedules = [[]]  # Empty schedule
        completed_jobs = set()
        job_assignments = {}
        job_to_family = {'J1': 'FAM1'}
        job_sequences = {'J1': 1}
        all_jobs = [job]
        
        valid, _ = rules.is_action_valid(
            job, machine, current_schedules, completed_jobs,
            job_assignments, job_to_family, job_sequences, all_jobs
        )
        assert valid, "Valid action marked as invalid"
    
    test.add_test("Rules engine validation", test_rules_engine)
    
    # Test 4: Reward function
    def test_reward_function():
        config = {
            'completion_reward': 1.0,
            'importance_bonus': 5.0,
            'urgency_multiplier': 10.0,
            'wait_penalty': 0.1,
            'makespan_penalty': 0.05
        }
        reward_fn = RewardFunction(config)
        
        job = {
            'processing_time': 2.0,
            'lcd_days_remaining': 5,
            'is_important': True
        }
        machine = {'machine_id': 0}
        machine_schedules = [[]]  # Empty schedules
        
        reward = reward_fn.calculate_step_reward(
            job=job,
            machine=machine,
            start_time=0,
            end_time=2,
            current_time=0,
            machine_schedules=machine_schedules,
            completed_jobs=0,
            total_jobs=10,
            makespan=2
        )
        assert isinstance(reward, (int, float)), "Reward should be numeric"
    
    test.add_test("Reward calculation", test_reward_function)
    
    return test.run()


def test_ppo_model():
    """Test PPO model components"""
    test = ComponentTest("PPO Model")
    
    # Test 1: State encoder
    def test_state_encoder():
        config = {
            'job_embedding_dim': 64,
            'machine_embedding_dim': 32,
            'hidden_dim': 128
        }
        encoder = StateEncoder(config)
        
        # Test with dummy observation
        obs = torch.randn(1, 100)  # Batch size 1, obs size 100
        encoded = encoder.encode_from_env_observation(obs, n_jobs=5, n_machines=3)
        
        assert 'job_embeddings' in encoded
        assert 'machine_embeddings' in encoded
    
    test.add_test("State encoder", test_state_encoder)
    
    # Test 2: Action masking
    def test_action_masking():
        masking = ActionMasking()
        
        # Test mask conversion
        env_mask = np.array([True, False, True, True, False])
        tensor_mask = masking.env_mask_to_tensor(env_mask)
        
        assert tensor_mask.shape == (1, 5)
        assert tensor_mask.dtype == torch.bool
    
    test.add_test("Action masking", test_action_masking)
    
    # Test 3: Transformer policy
    def test_transformer_policy():
        config = {
            'embed_dim': 256,  # Match the default embed_dim
            'hidden_dim': 128,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1,
            'max_jobs': 100,
            'max_machines': 50
        }
        policy = TransformerSchedulingPolicy(config)
        
        # Create test input with correct keys and dimensions
        encoded_state = {
            'job_embeddings': torch.randn(1, 5, 256),  # Use embed_dim
            'machine_embeddings': torch.randn(1, 3, 256),  # Use embed_dim
            'global_embedding': torch.randn(1, 256),  # Use embed_dim
            'job_mask': torch.ones(1, 5).bool(),
            'machine_mask': torch.ones(1, 3).bool()
        }
        action_mask = torch.ones(1, 15).bool()
        
        # Test forward pass
        logits, values = policy(encoded_state, action_mask)
        assert logits.shape == (1, 15)
        assert values.shape == (1, 1)
    
    test.add_test("Transformer policy", test_transformer_policy)
    
    # Test 4: Rollout buffer
    def test_rollout_buffer():
        buffer = RolloutBuffer(
            buffer_size=10,
            observation_shape=(50,),
            action_shape=(),
            n_envs=1,
            gamma=0.99,
            gae_lambda=0.95
        )
        
        # Add some data
        for i in range(5):
            buffer.add(
                obs=np.random.randn(1, 50),
                action=np.array([i]),
                reward=np.array([0.1]),
                done=np.array([False]),
                value=np.array([1.0]),
                log_prob=np.array([-0.5])
            )
        
        assert buffer.pos == 5
    
    test.add_test("Rollout buffer", test_rollout_buffer)
    
    # Test 5: Curriculum manager
    def test_curriculum():
        manager = CurriculumManager()
        
        stage = manager.get_current_stage()
        assert stage.name == "toy", f"Expected toy stage, got {stage.name}"
        
        env_config = manager.get_env_config()
        assert 'curriculum_stage' in env_config
    
    test.add_test("Curriculum manager", test_curriculum)
    
    # Test 6: PPO scheduler
    def test_ppo_scheduler():
        config = {
            'model': {
                'job_embedding_dim': 32,
                'machine_embedding_dim': 16,
                'hidden_dim': 64,
                'n_heads': 2,
                'n_layers': 1,
                'dropout': 0.0,
                'max_jobs': 10,
                'max_machines': 5
            },
            'ppo': {
                'clip_range': 0.2,
                'learning_rate': 3e-4
            },
            'device': 'cpu'
        }
        
        model = PPOScheduler(config)
        assert hasattr(model, 'state_encoder')
        assert hasattr(model, 'policy')
        assert hasattr(model, 'optimizer')
    
    test.add_test("PPO scheduler initialization", test_ppo_scheduler)
    
    return test.run()


def test_integration():
    """Test component integration"""
    test = ComponentTest("Integration")
    
    # Test 1: Data to environment flow
    def test_data_to_env():
        # Create test data - for single machine job
        jobs = [{
            "job_id": "INT001",
            "family_id": "FAM1",
            "sequence": 1,
            "required_machines": [100],  # Machine ID that will be found by db_machine_id
            "processing_time": 1.5,
            "lcd_days_remaining": 3,
            "is_important": False
        }]
        machines = [{
            "machine_id": 0,  # Index
            "db_machine_id": 100,  # Database ID
            "machine_name": "M0",
            "machine_type_id": 1
        }]
        
        # Create environment
        env = SchedulingGameEnv(jobs, machines, None, {})
        obs, _ = env.reset()
        
        # Check action space
        mask = env.get_action_mask()
        assert np.any(mask), "No valid actions available"
    
    test.add_test("Data → Environment flow", test_data_to_env)
    
    # Test 2: Environment to model flow
    def test_env_to_model():
        # Simple environment
        jobs = [{
            "job_id": "INT002",
            "family_id": "FAM1",
            "sequence": 1,
            "required_machines": [1],
            "processing_time": 2.0,
            "lcd_days_remaining": 5,
            "is_important": True
        }]
        machines = [{
            "machine_id": 1,
            "machine_name": "M1",
            "machine_type_id": 1
        }]
        
        env = SchedulingGameEnv(jobs, machines, None, {})
        obs, _ = env.reset()
        
        # Check observation is valid for model
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
    
    test.add_test("Environment → Model flow", test_env_to_model)
    
    return test.run()


def generate_report(results: List[Dict], output_path: str):
    """Generate comprehensive test report"""
    report = []
    report.append("# PPO Scheduling System - Comprehensive Test Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Executive Summary\n")
    
    # Overall statistics
    total_tests = sum(r['total_tests'] for r in results)
    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    report.append(f"- **Total Tests:** {total_tests}")
    report.append(f"- **Passed:** {total_passed}")
    report.append(f"- **Failed:** {total_failed}")
    report.append(f"- **Success Rate:** {overall_rate:.1f}%")
    report.append(f"- **Total Duration:** {sum(r['duration'] for r in results):.2f}s")
    
    # Component summary
    report.append("\n## Component Test Results\n")
    
    for result in results:
        report.append(f"### {result['component']}")
        report.append(f"- Tests: {result['total_tests']}")
        report.append(f"- Passed: {result['passed']}")
        report.append(f"- Failed: {result['failed']}")
        report.append(f"- Success Rate: {result['success_rate']:.1f}%")
        report.append(f"- Duration: {result['duration']:.2f}s")
        
        if result['failed'] > 0:
            report.append("\n**Failed Tests:**")
            for test_result in result['results']:
                if test_result['status'] == 'FAILED':
                    report.append(f"- {test_result['test']}: {test_result['error']}")
        
        report.append("")
    
    # System readiness
    report.append("\n## System Readiness Assessment\n")
    
    if overall_rate == 100:
        report.append("✅ **SYSTEM READY FOR TRAINING**")
        report.append("\nAll components tested successfully. The system is ready for Phase 3.")
    elif overall_rate >= 80:
        report.append("⚠️ **SYSTEM MOSTLY READY**")
        report.append("\nMost components working. Minor fixes needed before training.")
    else:
        report.append("❌ **SYSTEM NOT READY**")
        report.append("\nSignificant issues detected. Fix failing tests before proceeding.")
    
    # Critical components check
    report.append("\n## Critical Component Status\n")
    
    critical_components = {
        "Data Layer": "Data loading and processing",
        "Environment": "Game rules and state management",
        "PPO Model": "Neural network and training logic",
        "Integration": "Component communication"
    }
    
    for component, description in critical_components.items():
        result = next((r for r in results if r['component'] == component), None)
        if result:
            status = "✅" if result['success_rate'] == 100 else "❌"
            report.append(f"{status} **{component}**: {description}")
            report.append(f"   - Success Rate: {result['success_rate']:.1f}%")
    
    # Detailed error log
    report.append("\n## Detailed Error Log\n")
    
    has_errors = False
    for result in results:
        for test_result in result['results']:
            if test_result['status'] == 'FAILED':
                has_errors = True
                report.append(f"### {result['component']} - {test_result['test']}")
                report.append(f"**Error:** {test_result['error']}")
                report.append("```")
                report.append(test_result['traceback'].strip())
                report.append("```\n")
    
    if not has_errors:
        report.append("No errors detected. All tests passed successfully.")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Also save JSON results
    json_path = output_path.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Report saved to: {output_path}")
    print(f"📊 JSON results saved to: {json_path}")


def main():
    """Run all tests and generate report"""
    print("="*80)
    print("PPO SCHEDULING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all test suites
    test_suites = [
        test_data_layer,
        test_environment,
        test_ppo_model,
        test_integration
    ]
    
    for test_suite in test_suites:
        try:
            result = test_suite()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            traceback.print_exc()
    
    # Generate report
    report_path = "/Users/carrickcheah/Project/ppo/app_2/phase2/test_result/comprehensive_report.md"
    generate_report(results, report_path)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = sum(r['total_tests'] for r in results)
    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()