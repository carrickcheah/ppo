"""
Test the unconstrained model to see actual scheduling decisions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import PPO
from src.environments.medium_env_unconstrained import MediumUnconstrainedSchedulingEnv
import numpy as np

def test_unconstrained_model():
    """Test and show detailed results."""
    print("="*60)
    print("TESTING UNCONSTRAINED PPO MODEL")
    print("="*60)
    
    # Load model and create environment
    model = PPO.load("./models/medium_unconstrained/final_model")
    env = MediumUnconstrainedSchedulingEnv(seed=42)
    
    # Run one episode with detailed logging
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    # Track scheduling decisions
    scheduling_decisions = []
    priority_violations = []
    
    print("\nScheduling decisions (first 20):")
    print("-"*60)
    
    while not done and steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        
        # Log decision before step
        if action < len(env.valid_actions) and steps < 20:
            family_id, _, task = env.valid_actions[action]
            family = env.families_data[family_id]
            
            # Check all available priorities
            available_priorities = [env.families_data[fid]['priority'] for fid, _, _ in env.valid_actions]
            best_priority = min(available_priorities)
            
            selected_priority = family['priority']
            urgency = env._calculate_urgency(family_id)
            
            print(f"Step {steps+1}: Selected {family_id}-{task['sequence']} "
                  f"(Priority {selected_priority}, {urgency:.0f} days to LCD)")
            
            if selected_priority > best_priority:
                print(f"  ⚠️  VIOLATION: Could have chosen Priority {best_priority}")
                priority_violations.append({
                    'step': steps,
                    'selected': selected_priority,
                    'best': best_priority,
                    'difference': selected_priority - best_priority,
                    'urgency': urgency
                })
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    # Summary
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    
    completed = sum(len(c) for c in env.completed_tasks.values())
    print(f"Total steps: {steps}")
    print(f"Tasks completed: {completed}/{env.n_jobs} ({completed/env.n_jobs*100:.1f}%)")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Makespan: {env.episode_makespan:.1f}h")
    print(f"Priority violations: {env.priority_violations}")
    print(f"Urgency wins: {env.urgency_wins}")
    
    # Analyze violations
    if priority_violations:
        print("\n" + "="*60)
        print("VIOLATION ANALYSIS")
        print("="*60)
        
        by_difference = {}
        urgent_violations = 0
        
        for v in priority_violations[:10]:  # First 10 violations
            diff = v['difference']
            by_difference[diff] = by_difference.get(diff, 0) + 1
            if v['urgency'] < 14:
                urgent_violations += 1
            
            print(f"Step {v['step']+1}: Chose Priority {v['selected']} "
                  f"over Priority {v['best']} (diff={diff})")
            if v['urgency'] < 14:
                print(f"  ✓ Job was urgent ({v['urgency']:.0f} days)")
        
        print(f"\nViolation summary:")
        for diff, count in sorted(by_difference.items()):
            print(f"  {diff} levels lower: {count} times")
        print(f"  Urgent jobs: {urgent_violations}/{len(priority_violations[:10])}")
    
    # Show final schedule quality
    total_work = sum(
        task['processing_time'] 
        for family in env.families_data.values()
        for task in family['tasks']
    )
    theoretical_min = total_work / env.n_machines
    efficiency = theoretical_min / env.episode_makespan if env.episode_makespan > 0 else 0
    
    print(f"\nEfficiency: {efficiency:.1%} (theoretical min: {theoretical_min:.1f}h)")
    print(f"Gap to optimal: {env.episode_makespan - theoretical_min:.1f}h")

if __name__ == "__main__":
    test_unconstrained_model()