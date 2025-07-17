#!/usr/bin/env python3
"""Test why we're getting few scheduled jobs."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv

def test_scheduling():
    """Test scheduling to see what's happening."""
    
    env = ScaledProductionEnv(
        n_machines=10,
        max_episode_steps=200,
        max_valid_actions=100,
        data_file='app/data/large_production_data.json',
        snapshot_file='app/data/production_snapshot_latest.json'
    )
    
    # Check initial state
    print(f"Total jobs in environment: {env.n_jobs}")
    print(f"Total families: {env.n_families}")
    print(f"Break constraints loaded: {len(env.break_constraints.breaks)} breaks")
    
    # Reset and try scheduling
    obs, info = env.reset()
    print(f"\nInitial valid actions: {len(env.valid_actions)}")
    
    scheduled_count = 0
    break_delays = 0
    
    for step in range(100):
        if len(env.valid_actions) == 0:
            print(f"\nNo more valid actions at step {step}")
            break
            
        # Take action
        obs, reward, terminated, truncated, info = env.step(0)
        scheduled_count += 1
        
        if 'break_delay' in info:
            break_delays += 1
            
        if step < 5:
            print(f"Step {step}: Scheduled {info['scheduled_job']} at time {info['start_time']:.1f}h")
        
        if terminated:
            print(f"\nEnvironment terminated at step {step}")
            break
        if truncated:
            print(f"\nEnvironment truncated at step {step}")
            break
    
    print(f"\nTotal scheduled: {scheduled_count}")
    print(f"Jobs with break delays: {break_delays}")
    print(f"Current time reached: {env.current_time:.1f}h")
    
    # Check completion status
    total_completed = sum(len(completed) for completed in env.completed_tasks.values())
    print(f"Total tasks completed: {total_completed}/{env.n_jobs}")

if __name__ == "__main__":
    test_scheduling()