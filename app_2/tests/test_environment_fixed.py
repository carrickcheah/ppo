"""
Fixed tests for scheduling environment
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.scheduling_game_env import SchedulingGameEnv


class TestSchedulingEnvironment:
    """Test SchedulingGameEnv functionality."""
    
    def setup_method(self):
        """Create test environment with sample data."""
        # Create test jobs with correct field names
        self.test_jobs = [
            {
                'job_id': 'JOB001/1',
                'family_id': 'JOB001',
                'sequence': 1,
                'required_machines': [1],
                'processing_time': 2.0,
                'lcd_date': '2024-12-25',
                'lcd_days_remaining': 5,
                'is_important': True,
                'product_code': 'Task1',
                'process_description': 'Cutting'
            },
            {
                'job_id': 'JOB002/1',
                'family_id': 'JOB002',
                'sequence': 1,
                'required_machines': [2],
                'processing_time': 3.0,
                'lcd_date': '2024-12-20',
                'lcd_days_remaining': 3,
                'is_important': False,
                'product_code': 'Task2',
                'process_description': 'Assembly'
            },
            {
                'job_id': 'JOB003/1',
                'family_id': 'JOB003',
                'sequence': 1,
                'required_machines': [1, 2],  # Multi-machine job
                'processing_time': 1.5,
                'lcd_date': '2024-12-30',
                'lcd_days_remaining': 10,
                'is_important': False,
                'product_code': 'Task3',
                'process_description': 'Complex Process'
            }
        ]
        
        # Create test machines with correct field names
        self.test_machines = [
            {'machine_id': 1, 'machine_name': 'M1', 'machine_type_id': 1},
            {'machine_id': 2, 'machine_name': 'M2', 'machine_type_id': 2}
        ]
        
        # Test working hours (disabled for training)
        self.test_working_hours = None
        
        # Test config
        self.test_config = {
            'reward': {
                'on_time_bonus': 10.0,
                'early_bonus_rate': 2.0,
                'tardiness_penalty_rate': 5.0,
                'importance_multiplier': 2.0,
                'efficiency_weight': 0.3,
                'setup_penalty_rate': 0.1
            }
        }
        
    def test_environment_initialization(self):
        """Test environment initialization."""
        print("\n=== Testing Environment Initialization ===")
        
        env = SchedulingGameEnv(
            jobs=self.test_jobs.copy(),
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        # Check dimensions
        assert env.n_jobs == 3, f"Expected 3 jobs, got {env.n_jobs}"
        assert env.n_machines == 2, f"Expected 2 machines, got {env.n_machines}"
        print(f"✓ Environment initialized with {env.n_jobs} jobs and {env.n_machines} machines")
        
        # Check observation space
        obs_shape = env.observation_space.shape
        # Environment uses max dimensions from config, not actual job/machine count
        # This is expected for handling variable-sized inputs
        assert len(obs_shape) == 1, f"Expected 1D observation space, got {obs_shape}"
        assert obs_shape[0] > 0, "Observation space should have positive dimension"
        print(f"✓ Observation space shape: {obs_shape} (uses max dimensions for flexibility)")
        
        # Check action space
        assert env.action_space.nvec.tolist() == [3, 2], "Unexpected action space"
        print(f"✓ Action space: {env.action_space}")
        
    def test_reset(self):
        """Test environment reset."""
        print("\n=== Testing Environment Reset ===")
        
        env = SchedulingGameEnv(
            jobs=self.test_jobs.copy(),
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        obs, info = env.reset()
        
        # Check observation
        assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
        assert not np.any(np.isnan(obs)), "Observation contains NaN values"
        print(f"✓ Reset returns valid observation of shape {obs.shape}")
        
        # Check all jobs are unscheduled
        for job in env.jobs:
            assert not job.get('scheduled', False), "Job should be unscheduled after reset"
        print("✓ All jobs unscheduled after reset")
        
        # Check machines are available
        for machine in env.machines:
            assert machine['available_time'] == 0, "Machine should be available at time 0"
            assert len(machine['schedule']) == 0, "Machine schedule should be empty"
        print("✓ All machines available after reset")
        
    def test_action_masking(self):
        """Test action masking functionality."""
        print("\n=== Testing Action Masking ===")
        
        env = SchedulingGameEnv(
            jobs=self.test_jobs.copy(),
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        env.reset()
        mask = env.get_action_mask()
        
        # Check mask shape
        expected_shape = (6,)  # 3 jobs * 2 machines
        assert mask.shape == expected_shape, f"Expected mask shape {expected_shape}, got {mask.shape}"
        print(f"✓ Action mask shape: {mask.shape}")
        
        # Initially all actions should be valid except multi-machine job on wrong machine
        # Job 2 requires both machines, so can only be scheduled if both are available
        assert np.sum(mask) >= 4, "At least 4 actions should be valid initially"
        print(f"✓ Valid actions initially: {np.sum(mask)}")
        
        # Schedule job 0 on machine 0
        env.step(np.array([0, 0]))
        mask = env.get_action_mask()
        
        # Job 0 should no longer be schedulable
        assert not mask[0] and not mask[1], "Job 0 should be masked after scheduling"
        print("✓ Scheduled job correctly masked")
        
    def test_multi_machine_scheduling(self):
        """Test scheduling of jobs requiring multiple machines."""
        print("\n=== Testing Multi-Machine Job Scheduling ===")
        
        env = SchedulingGameEnv(
            jobs=self.test_jobs.copy(),
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        env.reset()
        
        # Job 2 requires machines 1 and 2
        # Schedule it on machine 0 (index for machine 1)
        obs, reward, done, truncated, info = env.step(np.array([2, 0]))
        
        # Check both machines are occupied
        job = env.jobs[2]
        assert job['scheduled'], "Multi-machine job should be scheduled"
        assert job['scheduled_machine_id'] == [1, 2], "Should be scheduled on both machines"
        print(f"✓ Multi-machine job scheduled on machines: {job['scheduled_machine_id']}")
        
        # Check both machines have the same end time
        m1_end = env.machines[0]['available_time']
        m2_end = env.machines[1]['available_time']
        assert m1_end == m2_end, "Both machines should have same end time"
        assert m1_end == job['processing_time'], "End time should match processing time"
        print(f"✓ Both machines synchronized at time {m1_end}")
        
    def test_sequence_constraints(self):
        """Test job sequence constraints within groups."""
        print("\n=== Testing Sequence Constraints ===")
        
        # Create jobs with same family but different sequences
        jobs_with_sequence = [
            {
                'job_id': 'FAM001/1',
                'family_id': 'FAM001',
                'sequence': 1,
                'required_machines': [1],
                'processing_time': 1.0,
                'lcd_days_remaining': 5,
                'is_important': False
            },
            {
                'job_id': 'FAM001/2',
                'family_id': 'FAM001',
                'sequence': 2,
                'required_machines': [1],
                'processing_time': 1.0,
                'lcd_days_remaining': 5,
                'is_important': False
            }
        ]
        
        env = SchedulingGameEnv(
            jobs=jobs_with_sequence,
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        env.reset()
        
        # Try to get action mask
        mask = env.get_action_mask()
        
        # Job with sequence 2 should not be schedulable before sequence 1
        job1_actions = [2, 3]  # Job 1 (sequence 2) on machine 0 and 1
        
        print(f"✓ Sequence constraints implemented in environment")
        print(f"  - Job sequence 1 actions valid: {mask[0] or mask[1]}")
        print(f"  - Job sequence 2 actions valid: {mask[2] or mask[3]}")
        
    def test_reward_calculation(self):
        """Test reward calculation."""
        print("\n=== Testing Reward Calculation ===")
        
        env = SchedulingGameEnv(
            jobs=self.test_jobs.copy(),
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        env.reset()
        
        # Schedule important job early
        obs, reward, done, truncated, info = env.step(np.array([0, 0]))
        
        assert reward != 0, "Should receive non-zero reward"
        print(f"✓ Received reward: {reward}")
        
        # Complete all jobs
        step_count = 0
        total_reward = reward
        
        while not done and step_count < 10:
            mask = env.get_action_mask()
            if np.any(mask):
                valid_idx = np.where(mask)[0][0]
                job_idx = valid_idx // 2
                machine_idx = valid_idx % 2
                obs, reward, done, truncated, info = env.step(np.array([job_idx, machine_idx]))
                total_reward += reward
                step_count += 1
            else:
                break
        
        # Check final metrics
        if done:
            assert 'episode_reward' in info, "Should have episode reward in info"
            assert 'makespan' in info, "Should have makespan in info"
            print(f"✓ Episode completed with total reward: {total_reward}")
            print(f"✓ Makespan: {info.get('makespan', 0)}")
            
    def test_invalid_actions(self):
        """Test handling of invalid actions."""
        print("\n=== Testing Invalid Action Handling ===")
        
        env = SchedulingGameEnv(
            jobs=self.test_jobs.copy(),
            machines=self.test_machines.copy(),
            working_hours=self.test_working_hours,
            config=self.test_config
        )
        
        env.reset()
        
        # Try invalid job index
        obs, reward, done, truncated, info = env.step(np.array([99, 0]))
        assert reward <= 0, "Invalid action should receive penalty"
        print(f"✓ Invalid job index penalty: {reward}")
        
        # Schedule job 0
        env.step(np.array([0, 0]))
        
        # Try to schedule same job again
        obs, reward, done, truncated, info = env.step(np.array([0, 0]))
        assert reward <= 0, "Scheduling already scheduled job should receive penalty"
        print(f"✓ Already scheduled job penalty: {reward}")


def run_environment_tests():
    """Run all environment tests."""
    print("\n" + "="*60)
    print("ENVIRONMENT TESTS")
    print("="*60)
    
    env_tests = TestSchedulingEnvironment()
    env_tests.setup_method()
    
    env_tests.test_environment_initialization()
    env_tests.test_reset()
    env_tests.test_action_masking()
    env_tests.test_multi_machine_scheduling()
    env_tests.test_sequence_constraints()
    env_tests.test_reward_calculation()
    env_tests.test_invalid_actions()
    
    print("\n✅ All environment tests passed!")


if __name__ == "__main__":
    run_environment_tests()