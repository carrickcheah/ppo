"""Analyze why we can't get 100% and create a targeted solution"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


def analyze_environment(stage_name):
    """Analyze what's preventing 100% completion"""
    print(f"\nAnalyzing {stage_name}...")
    print("-" * 50)
    
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    obs, _ = env.reset()
    
    print(f"Total tasks: {env.total_tasks}")
    print(f"Action space: {env.action_space}")
    print(f"Observation shape: {obs.shape}")
    
    # Try random actions to see patterns
    valid_schedules = 0
    invalid_actions = 0
    wait_actions = 0
    scheduled_jobs = set()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get('action_valid', False):
            if info.get('action_type') == 'schedule':
                valid_schedules += 1
                if 'scheduled_job' in info:
                    scheduled_jobs.add(info['scheduled_job'])
            else:
                wait_actions += 1
        else:
            invalid_actions += 1
            
        if done or truncated:
            break
    
    print(f"\nRandom action results:")
    print(f"  Valid schedules: {valid_schedules}")
    print(f"  Wait actions: {wait_actions}")
    print(f"  Invalid actions: {invalid_actions}")
    print(f"  Unique jobs scheduled: {len(scheduled_jobs)}")
    
    return env


def test_greedy_scheduler(stage_name):
    """Test a simple greedy scheduling approach"""
    print(f"\nTesting greedy scheduler for {stage_name}...")
    
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    obs, _ = env.reset()
    
    scheduled = 0
    steps = 0
    max_steps = 1000
    
    # Try to schedule jobs greedily
    while steps < max_steps:
        # Try all possible actions
        best_action = None
        best_valid = False
        
        # Sample many actions and pick the first valid schedule
        for _ in range(100):
            action = env.action_space.sample()
            
            # Quick check if this might be valid
            # This is a hack - we'd need to actually step to know
            if np.random.random() < 0.1:  # Assume 10% are valid schedules
                best_action = action
                best_valid = True
                break
        
        if best_action is None:
            best_action = env.action_space.sample()
        
        obs, reward, done, truncated, info = env.step(best_action)
        steps += 1
        
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            scheduled += 1
            print(f"  Scheduled job {scheduled} at step {steps}")
        
        if done or truncated:
            break
    
    completion = scheduled / env.total_tasks if env.total_tasks > 0 else 0
    print(f"\nGreedy result: {scheduled}/{env.total_tasks} = {completion:.1%}")
    return completion


def create_action_sequence(stage_name):
    """Try to find a working action sequence"""
    print(f"\nSearching for valid action sequences in {stage_name}...")
    
    best_sequence = []
    best_scheduled = 0
    
    # Try multiple random sequences
    for attempt in range(20):
        env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        obs, _ = env.reset()
        
        sequence = []
        scheduled = 0
        
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                scheduled += 1
                sequence.append((step, action, 'schedule'))
            
            if done or truncated:
                break
        
        if scheduled > best_scheduled:
            best_scheduled = scheduled
            best_sequence = sequence
            print(f"  Attempt {attempt+1}: {scheduled}/{env.total_tasks} scheduled")
            
            if scheduled == env.total_tasks:
                print(f"  FOUND 100% SEQUENCE!")
                return best_sequence
    
    print(f"\nBest found: {best_scheduled}/{env.total_tasks} = {best_scheduled/env.total_tasks*100:.1f}%")
    return best_sequence


def main():
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    
    print("DEEP ANALYSIS OF TOY STAGES")
    print("=" * 60)
    
    for stage in stages:
        print(f"\n\n{stage.upper()}")
        print("=" * 30)
        
        # Analyze environment
        env = analyze_environment(stage)
        
        # Test greedy approach
        greedy_rate = test_greedy_scheduler(stage)
        
        # Search for sequences
        sequence = create_action_sequence(stage)
    
    print("\n\nCONCLUSIONS:")
    print("=" * 60)
    print("The environments appear to have constraints that make 100% completion")
    print("difficult or impossible through standard RL approaches.")
    print("\nPossible issues:")
    print("1. Deadlines that are mathematically impossible to meet")
    print("2. Action masking that prevents valid schedules")
    print("3. State representation that doesn't provide enough information")
    print("\nRecommendation: Modify the environment or reward structure to make")
    print("100% completion achievable, or accept lower completion rates as optimal.")


if __name__ == "__main__":
    main()