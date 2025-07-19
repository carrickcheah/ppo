#!/usr/bin/env python3
"""
Test break time constraints implementation.
Verifies that jobs cannot be scheduled during break times.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv
from src.environments.break_time_constraints import BreakTimeConstraints
from datetime import datetime, time, timedelta
import numpy as np

def test_break_constraints():
    """Test the break time constraints functionality."""
    print("Testing Break Time Constraints")
    print("="*50)
    
    # Initialize break constraints
    constraints = BreakTimeConstraints()
    
    # Display loaded breaks
    print("\nLoaded break times:")
    for b in constraints.breaks:
        day_desc = "Every day" if b.day_of_week is None else f"Day {b.day_of_week}"
        print(f"  {b.name}: {b.start_time}-{b.end_time} ({day_desc})")
    
    # Test scenarios
    print("\n" + "="*50)
    print("Testing time slot validation:")
    
    # Test 1: Job during machine off time (should be invalid)
    print("\n1. Job at 2:00 AM (during Machine Off 1:00-6:30):")
    test_dt = datetime.now().replace(hour=2, minute=0, second=0, microsecond=0)
    is_valid = constraints.is_valid_time_slot(test_dt, 1.0)  # 1 hour job
    print(f"   Start: {test_dt.strftime('%H:%M')}, Duration: 1h")
    print(f"   Valid: {is_valid} (Should be False)")
    
    # Test 2: Job at 7:00 AM (after Machine Off)
    print("\n2. Job at 7:00 AM (after Machine Off):")
    test_dt = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
    is_valid = constraints.is_valid_time_slot(test_dt, 1.0)
    print(f"   Start: {test_dt.strftime('%H:%M')}, Duration: 1h")
    print(f"   Valid: {is_valid} (Should be True)")
    
    # Test 3: Job that would overlap lunch break
    print("\n3. Job at 12:00 that would overlap lunch (12:45-13:30):")
    test_dt = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    is_valid = constraints.is_valid_time_slot(test_dt, 1.5)  # 1.5 hour job
    print(f"   Start: {test_dt.strftime('%H:%M')}, Duration: 1.5h")
    print(f"   Valid: {is_valid} (Should be False)")
    
    # Test 4: Get next valid start time
    print("\n4. Finding next valid start for job at 12:00:")
    next_valid = constraints.get_next_valid_start(test_dt, 1.5)
    print(f"   Original: {test_dt.strftime('%H:%M')}")
    print(f"   Next valid: {next_valid.strftime('%H:%M')} (Should be after 13:30)")
    
    # Test 5: Sunday scheduling
    print("\n5. Job on Sunday:")
    # Find next Sunday
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0:
        days_until_sunday = 7
    sunday = today + timedelta(days=days_until_sunday)
    sunday = sunday.replace(hour=10, minute=0, second=0, microsecond=0)
    is_valid = constraints.is_valid_time_slot(sunday, 2.0)
    print(f"   Date: {sunday.strftime('%A, %Y-%m-%d %H:%M')}")
    print(f"   Valid: {is_valid} (Should be False - Sunday Off)")
    
    # Test 6: Saturday afternoon
    print("\n6. Job on Saturday afternoon:")
    saturday = sunday - timedelta(days=1)
    saturday = saturday.replace(hour=14, minute=0, second=0, microsecond=0)
    is_valid = constraints.is_valid_time_slot(saturday, 2.0)
    print(f"   Date: {saturday.strftime('%A, %Y-%m-%d %H:%M')}")
    print(f"   Valid: {is_valid} (Should be False - Saturday Half Day)")

def test_environment_integration():
    """Test break constraints in the full environment."""
    print("\n" + "="*50)
    print("Testing Environment Integration:")
    
    # Initialize environment
    env = ScaledProductionEnv(
        n_machines=10,
        max_episode_steps=50,
        max_valid_actions=10,
        data_file='app/data/large_production_data.json',
        snapshot_file='app/data/production_snapshot_latest.json'
    )
    
    # Reset environment
    obs, info = env.reset()
    
    print("\nScheduling jobs with break constraints...")
    print(f"Base date: {env.base_date.strftime('%Y-%m-%d')}")
    
    # Take a few steps
    for step in range(5):
        # Get valid actions
        if len(env.valid_actions) == 0:
            print(f"\nStep {step}: No valid actions available")
            break
            
        # Take first valid action
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step}:")
        print(f"  Scheduled: {info['scheduled_job']}")
        print(f"  Start time: {info['start_time']:.2f}h")
        print(f"  End time: {info['end_time']:.2f}h")
        
        # Convert to actual time for display
        start_dt = env.break_constraints.hours_to_datetime(info['start_time'], env.base_date)
        end_dt = env.break_constraints.hours_to_datetime(info['end_time'], env.base_date)
        print(f"  Actual time: {start_dt.strftime('%a %H:%M')} - {end_dt.strftime('%a %H:%M')}")
        
        if 'break_delay' in info:
            print(f"  Break delay: {info['break_delay']:.2f}h")
        
        if terminated or truncated:
            break

def main():
    """Run all tests."""
    test_break_constraints()
    test_environment_integration()
    print("\n" + "="*50)
    print("Break constraint tests completed!")

if __name__ == "__main__":
    main()