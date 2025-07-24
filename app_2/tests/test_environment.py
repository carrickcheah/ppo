"""
Test the scheduling game environment.

This script tests basic environment functionality with toy data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from pathlib import Path

from src.environment.scheduling_game_env import SchedulingGameEnv
from src.environment.rules_engine import RulesEngine
from src.environment.reward_function import RewardFunction
from src.data.data_loader import DataLoader


def test_environment_setup():
    """Test basic environment setup and reset."""
    print("Testing environment setup...")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "environment.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use test data
    config['data']['source'] = 'test'
    
    # Create data loader
    data_loader = DataLoader(config['data'])
    
    # Load test data
    jobs = data_loader.load_jobs()
    machines = data_loader.load_machines()
    working_hours = data_loader.load_working_hours()
    
    print(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
    
    # Create environment
    env = SchedulingGameEnv(
        jobs=jobs,
        machines=machines,
        working_hours=working_hours,
        config=config
    )
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    return env, jobs, machines


def test_valid_action():
    """Test executing a valid action."""
    print("\nTesting valid action...")
    
    env, jobs, machines = test_environment_setup()
    obs, _ = env.reset()
    
    # Get action mask to find valid actions
    action_mask = env.get_action_mask()
    valid_actions = np.where(action_mask)[0]
    
    if len(valid_actions) > 0:
        # Take first valid action
        action_flat = valid_actions[0]
        
        # Convert flat action to (job, machine)
        n_jobs = len(jobs)
        job_idx = action_flat // len(machines)
        machine_idx = action_flat % len(machines)
        action = np.array([job_idx, machine_idx])
        
        print(f"Taking action: Job {job_idx} on Machine {machine_idx}")
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
    else:
        print("No valid actions available!")


def test_sequence_constraint():
    """Test that sequence constraints are enforced."""
    print("\nTesting sequence constraints...")
    
    env, jobs, machines = test_environment_setup()
    obs, _ = env.reset()
    
    # Try to schedule job with sequence 2 before sequence 1
    # Find a job with sequence > 1
    second_seq_job = None
    for idx, job in enumerate(jobs):
        if job.get('sequence', 1) == 2:
            second_seq_job = idx
            break
    
    if second_seq_job is not None:
        # Try to schedule it
        action = np.array([second_seq_job, 0])  # Try on first machine
        
        print(f"Trying to schedule sequence 2 job before sequence 1...")
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('invalid_action'):
            print(f"Good! Action was rejected: {info.get('reason')}")
            print(f"Penalty received: {reward}")
        else:
            print("ERROR: Sequence constraint not enforced!")
    else:
        print("No sequence 2 job found in test data")


def test_complete_episode():
    """Test completing a full episode."""
    print("\nTesting complete episode...")
    
    env, jobs, machines = test_environment_setup()
    obs, _ = env.reset()
    
    total_reward = 0
    step_count = 0
    max_steps = 100
    
    while step_count < max_steps:
        # Get valid actions
        action_mask = env.get_action_mask()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            print("No more valid actions!")
            break
        
        # Take first valid action (in real training, policy would choose)
        action_flat = valid_actions[0]
        n_jobs = len(jobs)
        job_idx = action_flat // len(machines)
        machine_idx = action_flat % len(machines)
        action = np.array([job_idx, machine_idx])
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if info.get('valid_action'):
            print(f"Step {step_count}: Scheduled {info['scheduled_job']} on {info['on_machine']}, "
                  f"Reward: {reward:.2f}")
        
        if terminated:
            print(f"\nEpisode completed!")
            print(f"Total steps: {step_count}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"All jobs scheduled: {info.get('completed_jobs')} / {len(jobs)}")
            break
    
    env.render()


def test_rules_engine():
    """Test the rules engine directly."""
    print("\nTesting rules engine...")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "environment.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create rules engine
    rules = RulesEngine(config['game']['rules'])
    
    # Create test scenario
    test_job = {
        'job_id': 'TEST_2/3',
        'machine_types': [1, 2, 3]
    }
    
    test_machine = {
        'machine_id': 0,
        'machine_type_id': 2
    }
    
    # Test compatibility check
    is_valid, reason = rules._check_compatibility_constraint(test_job, test_machine)
    print(f"Compatibility check: {is_valid} - {reason}")
    
    # Test with incompatible machine
    test_machine_bad = {
        'machine_id': 1,
        'machine_type_id': 5
    }
    
    is_valid, reason = rules._check_compatibility_constraint(test_job, test_machine_bad)
    print(f"Incompatibility check: {is_valid} - {reason}")


def test_reward_function():
    """Test the reward function calculations."""
    print("\nTesting reward function...")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "environment.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create reward function
    reward_fn = RewardFunction(config['rewards'])
    
    # Test scenarios
    
    # Scenario 1: Important job with high urgency
    important_urgent_job = {
        'job_id': 'TEST001',
        'is_important': True,
        'lcd_days_remaining': 2
    }
    
    machine = {'machine_id': 0, 'machine_name': 'M01'}
    
    reward = reward_fn.calculate_step_reward(
        job=important_urgent_job,
        machine=machine,
        start_time=0,
        end_time=2,
        current_time=0,
        machine_schedules=[[], [], []],
        completed_jobs=5,
        total_jobs=20,
        makespan=10
    )
    
    print(f"Important + Urgent job reward: {reward:.2f}")
    
    # Scenario 2: Regular job with low urgency
    regular_job = {
        'job_id': 'TEST002',
        'is_important': False,
        'lcd_days_remaining': 25
    }
    
    reward = reward_fn.calculate_step_reward(
        job=regular_job,
        machine=machine,
        start_time=0,
        end_time=2,
        current_time=0,
        machine_schedules=[[], [], []],
        completed_jobs=5,
        total_jobs=20,
        makespan=10
    )
    
    print(f"Regular job reward: {reward:.2f}")
    
    # Get reward breakdown
    info = reward_fn.get_reward_info(reward, regular_job, machine)
    print(f"Reward breakdown: {info['reward_components']}")


if __name__ == "__main__":
    print("=== PPO Scheduling Environment Tests ===\n")
    
    # Run all tests
    test_environment_setup()
    test_valid_action()
    test_sequence_constraint()
    test_complete_episode()
    test_rules_engine()
    test_reward_function()
    
    print("\n=== All tests completed! ===")
    print("\nThe environment is ready for training the PPO model.")
    print("Next steps:")
    print("1. Implement the PPO model architecture (transformer policy)")
    print("2. Create the training loop")
    print("3. Run curriculum learning from toy to production scale")
    print("4. Deploy the trained model via API")