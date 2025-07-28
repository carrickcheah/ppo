"""
Test all toy stage models and generate schedules for visualization
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal

def test_toy_model(stage_name):
    """Test a toy model and generate schedule"""
    print(f"\nTesting {stage_name}:")
    print("-" * 50)
    
    # Load model
    model_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/{stage_name}/final_model.zip"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    model = PPO.load(model_path)
    
    # Create environment
    env = CurriculumEnvironmentReal(stage_name=stage_name, verbose=False)
    
    # Run episode to generate schedule
    obs, _ = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        if info.get('action_taken', False):
            job_idx = info.get('job_index', -1)
            machine_idx = info.get('machine_index', -1)
            if job_idx >= 0:
                job = env.pending_jobs[job_idx]
                machine = env.machines[machine_idx]
                print(f"Step {step_count}: Scheduled {job['job_id']} on machine {machine['machine_id']}")
    
    # Get final schedule
    schedule = env.get_schedule()
    
    # Calculate metrics
    total_jobs = len(env.all_jobs)
    scheduled_jobs = len(schedule)
    completion_rate = scheduled_jobs / total_jobs if total_jobs > 0 else 0
    
    print(f"\nScheduled {scheduled_jobs}/{total_jobs} jobs ({completion_rate:.1%})")
    
    # Check sequence violations
    family_sequences = {}
    for task in schedule:
        family_id = task['family_id']
        sequence = task['sequence']
        if family_id not in family_sequences:
            family_sequences[family_id] = []
        family_sequences[family_id].append((sequence, task['start_time']))
    
    violations = 0
    for family_id, sequences in family_sequences.items():
        sequences.sort(key=lambda x: x[0])  # Sort by sequence number
        for i in range(1, len(sequences)):
            if sequences[i][1] < sequences[i-1][1]:  # Later sequence starts before earlier
                violations += 1
                print(f"Sequence violation in family {family_id}")
    
    print(f"Sequence violations: {violations}")
    
    # Save schedule
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/schedules"
    os.makedirs(output_dir, exist_ok=True)
    
    schedule_data = {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'total_jobs': total_jobs,
            'scheduled_jobs': scheduled_jobs,
            'completion_rate': completion_rate,
            'sequence_violations': violations,
            'total_reward': env.episode_reward
        },
        'schedule': schedule,
        'machines': [{'machine_id': m['machine_id'], 'machine_name': m['machine_name']} for m in env.machines]
    }
    
    output_path = os.path.join(output_dir, f"{stage_name}_schedule.json")
    with open(output_path, 'w') as f:
        json.dump(schedule_data, f, indent=2)
    
    print(f"Schedule saved to {output_path}")
    
    return schedule_data

def main():
    """Test all toy stages"""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Testing Toy Stage Models")
    print("=" * 60)
    
    all_results = {}
    
    for stage in stages:
        try:
            result = test_toy_model(stage)
            if result:
                all_results[stage] = result
        except Exception as e:
            print(f"Error testing {stage}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 60)
    
    for stage, result in all_results.items():
        metrics = result['metrics']
        print(f"\n{stage}:")
        print(f"  - Scheduled: {metrics['scheduled_jobs']}/{metrics['total_jobs']} ({metrics['completion_rate']:.1%})")
        print(f"  - Sequence violations: {metrics['sequence_violations']}")
        print(f"  - Total reward: {metrics['total_reward']:.2f}")

if __name__ == "__main__":
    main()