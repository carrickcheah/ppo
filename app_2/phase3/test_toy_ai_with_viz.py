"""
Test toy models and generate schedules for visualization
Uses the correct models from truly_fixed_models directory
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

def test_and_save_schedule(stage_name):
    """Test a model and save its schedule for visualization"""
    print(f"\nTesting {stage_name}:")
    print("-" * 50)
    
    # Check multiple model locations
    model_paths = [
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models/{stage_name}_final.zip",
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/{stage_name}/final_model.zip",
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/truly_fixed/{stage_name}/final_model.zip"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"No model found for {stage_name}")
        return None
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    # Run one episode to generate schedule
    obs, _ = env.reset()
    done = False
    step_count = 0
    max_steps = 200
    episode_reward = 0
    
    schedule = []
    
    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        episode_reward += reward
        step_count += 1
        
        # Check if a job was scheduled
        if hasattr(env, 'job_assignments') and len(env.job_assignments) > len(schedule):
            # New job was scheduled
            for job_key, job_info in env.job_assignments.items():
                if not any(s['job_id'] == job_key for s in schedule):
                    # Extract family and sequence
                    if '_seq' in job_key:
                        parts = job_key.split('_seq')
                        family_id = parts[0]
                        sequence = parts[1] if len(parts) > 1 else "1/1"
                    else:
                        family_id = job_key
                        sequence = "1/1"
                    
                    # Get deadline info
                    lcd_hours = 7 * 24  # Default 7 days
                    is_important = False
                    if hasattr(env, 'families') and family_id in env.families:
                        family_data = env.families[family_id]
                        lcd_hours = family_data.get('lcd_days_remaining', 7) * 24
                        is_important = family_data.get('is_important', False)
                    
                    schedule.append({
                        'job_id': job_key,
                        'family_id': family_id,
                        'sequence': sequence,
                        'machine_id': job_info.get('machine', 0),
                        'machine_name': f"M{job_info.get('machine', 0)}",
                        'start_time': job_info.get('start', 0),
                        'processing_time': job_info.get('processing_time', job_info.get('end', 0) - job_info.get('start', 0)),
                        'end_time': job_info.get('end', 0),
                        'lcd_date': lcd_hours,
                        'lcd_hours': lcd_hours,
                        'is_important': is_important,
                        'assigned_machines': [job_info.get('machine', 0)]
                    })
    
    # Get final metrics
    total_jobs = env.total_tasks
    scheduled_jobs = len(env.scheduled_jobs) if hasattr(env, 'scheduled_jobs') else len(schedule)
    completion_rate = scheduled_jobs / total_jobs if total_jobs > 0 else 0
    
    print(f"AI scheduled {scheduled_jobs}/{total_jobs} jobs ({completion_rate:.1%})")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Steps taken: {step_count}")
    
    # Get machine info
    machines = []
    if hasattr(env, 'n_machines'):
        n_machines = env.n_machines
    else:
        # Infer from schedule
        n_machines = max(job['machine_id'] for job in schedule) + 1 if schedule else 3
    
    for i in range(n_machines):
        machines.append({
            'machine_id': i,
            'machine_name': f'Machine {i}'
        })
    
    # Check sequence violations
    violations = 0
    family_jobs = {}
    for job in schedule:
        fam = job['family_id']
        if fam not in family_jobs:
            family_jobs[fam] = []
        family_jobs[fam].append(job)
    
    for fam, jobs in family_jobs.items():
        jobs.sort(key=lambda x: int(x['sequence'].split('/')[0]))
        for i in range(1, len(jobs)):
            if jobs[i]['start_time'] < jobs[i-1]['end_time']:
                violations += 1
    
    print(f"Sequence violations: {violations}")
    
    return {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'metrics': {
            'total_jobs': total_jobs,
            'scheduled_jobs': scheduled_jobs,
            'completion_rate': completion_rate,
            'sequence_violations': violations,
            'total_reward': episode_reward,
            'description': get_stage_description(stage_name)
        },
        'schedule': schedule,
        'machines': machines
    }

def get_stage_description(stage_name):
    descriptions = {
        'toy_easy': 'Learn sequence rules',
        'toy_normal': 'Learn deadlines',
        'toy_hard': 'Learn priorities',
        'toy_multi': 'Learn multi-machine'
    }
    return descriptions.get(stage_name, '')

def main():
    """Test all toy stages and save AI-generated schedules"""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Testing Toy Models and Generating AI Schedules")
    print("=" * 60)
    
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/ai_schedules"
    os.makedirs(output_dir, exist_ok=True)
    
    for stage in stages:
        try:
            result = test_and_save_schedule(stage)
            if result:
                # Save schedule
                output_path = os.path.join(output_dir, f"{stage}_ai_schedule.json")
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved AI schedule to: {output_path}")
        except Exception as e:
            print(f"Error with {stage}: {e}")
    
    print("\n" + "=" * 60)
    print("All AI schedules generated!")
    print("Run visualize_toy_schedules.py to create Gantt charts")

if __name__ == "__main__":
    main()