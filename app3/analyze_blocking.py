"""
Analyze why training gets stuck and propose solutions.
"""

import json
from pathlib import Path
from src.data.snapshot_loader import SnapshotLoader
from src.environments.scheduling_env import SchedulingEnv

def analyze_episode_completion():
    """Analyze why episodes don't complete all tasks."""
    
    # Test with 20 jobs (where training failed)
    data_file = Path("data/20_jobs.json")
    loader = SnapshotLoader(data_file)
    
    print(f"Data Analysis for {data_file.name}")
    print(f"Total tasks: {len(loader.tasks)}")
    print(f"Total families: {len(loader.families)}")
    print()
    
    # Check sequence constraints
    sequence_blocked = 0
    material_blocked = 0
    
    for task in loader.tasks:
        family = loader.families[task.family_id]
        
        # Check if task requires predecessors
        if task.sequence > 1:
            sequence_blocked += 1
            
        # Check material arrival
        try:
            if float(family.material_arrival) > 0:
                material_blocked += 1
        except (ValueError, TypeError):
            pass  # Skip if not a valid number
            material_blocked += 1
    
    print(f"Tasks blocked by sequence (seq > 1): {sequence_blocked}/{len(loader.tasks)} ({sequence_blocked/len(loader.tasks)*100:.1f}%)")
    print(f"Tasks blocked by material arrival: {material_blocked}/{len(loader.tasks)} ({material_blocked/len(loader.tasks)*100:.1f}%)")
    print()
    
    # Simulate a random policy
    env = SchedulingEnv(data_file)
    obs, info = env.reset()
    
    episode_done = False
    steps = 0
    max_steps = 1000
    
    while not episode_done and steps < max_steps:
        # Get valid actions
        valid_actions = [i for i, v in enumerate(info['action_mask']) if v]
        
        if not valid_actions:
            # No valid actions, advance time
            obs, reward, terminated, truncated, info = env.step(0)
        else:
            # Take first valid action (simple policy)
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
        
        steps += 1
        episode_done = terminated or truncated
        
        if steps % 100 == 0:
            print(f"Step {steps}: Scheduled {info['tasks_scheduled']}/{info['total_tasks']} tasks")
    
    print(f"\nFinal result after {steps} steps:")
    print(f"Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    print(f"Episode terminated: {terminated}")
    print(f"Episode truncated: {truncated}")
    
    return info['tasks_scheduled'] == info['total_tasks']

def propose_solutions():
    """Propose solutions based on analysis."""
    
    print("\n" + "="*50)
    print("IDENTIFIED ISSUES:")
    print("="*50)
    
    print("""
1. SUCCESS CRITERIA TOO STRICT:
   - Current: Success = 100% of tasks scheduled
   - Problem: With sequence constraints, this may be impossible within max_steps
   - Many tasks blocked waiting for predecessors
   
2. EPISODE TERMINATION:
   - Current: Episode ends only when ALL tasks scheduled
   - Problem: If stuck due to sequences, episode never ends naturally
   - Relies on truncation at max_steps (default 1000)
   
3. REWARD STRUCTURE:
   - Current: Small rewards (+5) per action, big penalties for late (-100/day)
   - Problem: Model gets positive rewards but never "succeeds"
   - No intermediate success metrics
   
4. CURRICULUM TOO AGGRESSIVE:
   - Stage 1→2: Jump from 34 to 65 tasks (91% increase)
   - Stage 2 requires 85% success immediately
   - No gradual learning of sequence handling
""")
    
    print("\n" + "="*50)
    print("RECOMMENDED SOLUTIONS:")
    print("="*50)
    
    print("""
1. REDEFINE SUCCESS CRITERIA:
   Option A: Partial success based on percentage
   - Success if >80% tasks scheduled
   - Gradual increase: 60%→70%→80%→90%
   
   Option B: Family-based success
   - Success if >70% of families have all tasks completed
   - Focuses on completing entire jobs
   
2. ADJUST EPISODE TERMINATION:
   - Add early termination when no progress for N steps
   - Consider episode done when no valid actions remain
   - Reduce max_steps to force faster learning
   
3. IMPROVE REWARD STRUCTURE:
   - Add sequence completion bonus (+50 per family completed)
   - Progressive rewards for hitting milestones (25%, 50%, 75% scheduled)
   - Reduce idle penalty to encourage waiting for sequences
   
4. GENTLER CURRICULUM:
   Stage 1: 10 jobs, 50% success threshold
   Stage 2: 20 jobs, 60% success threshold  
   Stage 3: 40 jobs, 65% success threshold
   Stage 4: 60 jobs, 70% success threshold
   Stage 5: 100 jobs, 75% success threshold
   Stage 6: 200+ jobs, 80% success threshold
   
5. TRAINING MODIFICATIONS:
   - Increase n_steps to 4096 for better GAE estimation
   - Use larger batch size (256) for more stable updates
   - Add curriculum warm-up: 10k steps at lower LR
""")

if __name__ == "__main__":
    # Analyze the problem
    success = analyze_episode_completion()
    
    # Propose solutions
    propose_solutions()
    
    print("\n" + "="*50)
    print("IMMEDIATE ACTIONS NEEDED:")
    print("="*50)
    print("""
1. Modify success criteria in curriculum_trainer.py
2. Add family completion tracking
3. Adjust stage thresholds
4. Update reward calculator with sequence bonuses
""")