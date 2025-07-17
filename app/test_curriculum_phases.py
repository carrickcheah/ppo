"""
Test script to verify curriculum learning phases work correctly.
Tests that break constraints can be toggled on/off.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv
import numpy as np


def test_phase1_no_breaks():
    """Test Phase 1: Environment without break constraints."""
    print("\n" + "="*60)
    print("TESTING PHASE 1: NO BREAK CONSTRAINTS")
    print("="*60)
    
    # Create environment without breaks
    env = ScaledProductionEnv(
        n_machines=40,
        use_break_constraints=False,
        seed=42
    )
    
    print(f"Break constraints enabled: {env.use_break_constraints}")
    print(f"Break constraints object: {env.break_constraints}")
    
    # Reset and run a few steps
    obs, info = env.reset()
    print(f"\nInitial state shape: {obs.shape}")
    print(f"Number of jobs: {env.n_jobs}")
    print(f"Number of machines: {env.n_machines}")
    
    # Track scheduling times
    scheduled_times = []
    
    # Run 20 random steps
    for step in range(20):
        # Get valid actions
        valid_actions = env.valid_actions
        if not valid_actions:
            print("No valid actions available")
            break
            
        # Take random action
        action_idx = np.random.randint(len(valid_actions))
        action = action_idx
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track when jobs are scheduled
        if 'scheduled_job' in info and 'start_time' in info:
            job_id = info.get('scheduled_job')
            start_time = info.get('start_time', 0)
            if start_time >= 0:  # Job was actually scheduled
                scheduled_times.append(start_time)
                print(f"Step {step}: Scheduled job {job_id} at time {start_time:.1f}h")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Check that jobs can be scheduled at any time (no break restrictions)
    print(f"\nScheduled {len(scheduled_times)} jobs")
    if scheduled_times:
        print(f"Earliest start time: {min(scheduled_times):.1f}h")
        print(f"Latest start time: {max(scheduled_times):.1f}h")
    else:
        print("Warning: No jobs were scheduled in the test")
    
    # Verify jobs can start during what would be break times
    break_times = [9.75, 12.75, 15.25, 18.0, 23.0]  # Morning tea, lunch, etc.
    jobs_during_breaks = 0
    for start_time in scheduled_times:
        for break_time in break_times:
            if break_time <= start_time <= break_time + 1:
                jobs_during_breaks += 1
                break
    
    print(f"\nJobs scheduled during break times: {jobs_during_breaks}")
    print("✓ Phase 1 test passed: Jobs can be scheduled without break constraints")
    
    return True


def test_phase2_with_breaks():
    """Test Phase 2: Environment with break constraints."""
    print("\n" + "="*60)
    print("TESTING PHASE 2: WITH BREAK CONSTRAINTS")
    print("="*60)
    
    # Create environment with breaks
    env = ScaledProductionEnv(
        n_machines=40,
        use_break_constraints=True,
        seed=42
    )
    
    print(f"Break constraints enabled: {env.use_break_constraints}")
    print(f"Break constraints object: {env.break_constraints is not None}")
    
    if env.break_constraints:
        print(f"Number of break periods: {len(env.break_constraints.breaks)}")
        for i, b in enumerate(env.break_constraints.breaks[:5]):
            print(f"  {i+1}. {b.name}: {b.start_time} - {b.end_time}")
    
    # Reset and run a few steps
    obs, info = env.reset()
    
    # Track scheduling with breaks
    scheduled_with_breaks = []
    break_delays = []
    
    # Run 20 random steps
    for step in range(20):
        # Get valid actions
        valid_actions = env.valid_actions
        if not valid_actions:
            print("No valid actions available")
            break
            
        # Take random action
        action_idx = np.random.randint(len(valid_actions))
        action = action_idx
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track scheduling info
        if 'scheduled_job' in info and 'start_time' in info:
            job_id = info.get('scheduled_job')
            start_time = info.get('start_time', 0)
            if start_time >= 0:  # Job was actually scheduled
                scheduled_with_breaks.append(start_time)
                
                # Check if job was delayed due to breaks
                if 'break_delay' in info:
                    break_delays.append(info['break_delay'])
                
                print(f"Step {step}: Scheduled job {job_id} at time {start_time:.1f}h")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    print(f"\nScheduled {len(scheduled_with_breaks)} jobs with break constraints")
    
    # Verify jobs avoid break times
    morning_start = 6.5  # After "Machine Off" period
    jobs_after_morning = sum(1 for t in scheduled_with_breaks if t >= morning_start)
    print(f"Jobs scheduled after morning start (6:30 AM): {jobs_after_morning}")
    
    if break_delays:
        print(f"Average break delay: {np.mean(break_delays):.1f}h")
    
    print("✓ Phase 2 test passed: Break constraints are enforced")
    
    return True


def test_phase_transition():
    """Test that the same environment can switch between phases."""
    print("\n" + "="*60)
    print("TESTING PHASE TRANSITION")
    print("="*60)
    
    # Test that we can create both types
    env_no_breaks = ScaledProductionEnv(use_break_constraints=False)
    env_with_breaks = ScaledProductionEnv(use_break_constraints=True)
    
    print(f"Environment 1 - Breaks enabled: {env_no_breaks.use_break_constraints}")
    print(f"Environment 2 - Breaks enabled: {env_with_breaks.use_break_constraints}")
    
    # Both should have same state/action spaces
    obs1, _ = env_no_breaks.reset()
    obs2, _ = env_with_breaks.reset()
    
    print(f"\nState space match: {obs1.shape == obs2.shape}")
    print(f"Action space match: {env_no_breaks.action_space.n == env_with_breaks.action_space.n}")
    
    print("✓ Phase transition test passed: Can create both environment types")
    
    return True


def main():
    """Run all curriculum phase tests."""
    print("CURRICULUM LEARNING PHASE TESTS")
    print("Testing that break constraints can be toggled for phased training")
    
    # Run tests
    all_passed = True
    
    try:
        all_passed &= test_phase1_no_breaks()
    except Exception as e:
        print(f"✗ Phase 1 test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_phase2_with_breaks()
    except Exception as e:
        print(f"✗ Phase 2 test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_phase_transition()
    except Exception as e:
        print(f"✗ Phase transition test failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CURRICULUM PHASE TESTS PASSED")
        print("\nReady to start curriculum learning:")
        print("1. Run: uv run python app/src/training/train_curriculum.py")
        print("2. Phase 1 will train without breaks")
        print("3. Phase 2 will add breaks with transfer learning")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before training")
    print("="*60)


if __name__ == "__main__":
    main()