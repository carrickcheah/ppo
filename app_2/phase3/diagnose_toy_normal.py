"""
Diagnose why toy_normal is stuck at low performance
Analyze the environment and reward structure
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

def analyze_toy_normal():
    """Analyze toy_normal environment and reward structure."""
    
    print("Analyzing toy_normal environment...")
    print("="*60)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed('toy_normal', verbose=True)
    
    # Reset and examine initial state
    obs, info = env.reset()
    
    print(f"\nEnvironment details:")
    print(f"  Total tasks: {env.total_tasks}")
    print(f"  Number of machines: {len(env.machines) if hasattr(env, 'machines') else 'N/A'}")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Analyze job deadlines
    print(f"\nJob deadline analysis:")
    families = getattr(env, 'families', {})
    if not families:
        print("  No families data available")
        return
    jobs_by_deadline = {}
    
    for family_id, family_data in families.items():
        lcd_days = family_data.get('lcd_days_remaining', 999)
        tasks = family_data.get('tasks', [])
        
        for task in tasks:
            processing_time = task.get('processing_time', 0)
            job_key = f"{family_id}_seq{task['sequence']}"
            
            jobs_by_deadline[job_key] = {
                'family': family_id,
                'sequence': task['sequence'],
                'processing_hours': processing_time,
                'deadline_days': lcd_days,
                'deadline_hours': lcd_days * 24,
                'is_important': family_data.get('is_important', False)
            }
    
    # Sort by deadline
    sorted_jobs = sorted(jobs_by_deadline.items(), key=lambda x: x[1]['deadline_hours'])
    
    print(f"\nJobs sorted by deadline urgency:")
    total_processing = 0
    for job_id, job_info in sorted_jobs[:10]:  # Show first 10
        total_processing += job_info['processing_hours']
        print(f"  {job_id}: {job_info['processing_hours']:.1f}h processing, "
              f"{job_info['deadline_hours']:.0f}h deadline "
              f"({'IMPORTANT' if job_info['is_important'] else 'normal'})")
    
    print(f"\nTotal processing time for all jobs: {sum(j['processing_hours'] for j in jobs_by_deadline.values()):.1f} hours")
    print(f"Tightest deadline: {sorted_jobs[0][1]['deadline_hours']:.0f} hours")
    
    # Test a few random actions
    print(f"\nTesting random actions to see rewards:")
    
    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get('action_valid', False):
            print(f"  Action {action}: Valid - Reward = {reward:.2f}")
            if 'scheduled_job' in info:
                print(f"    Scheduled: {info['scheduled_job']}")
        else:
            print(f"  Action {action}: Invalid - Reward = {reward:.2f}")
    
    # Test with existing model if available
    model_paths = [
        "/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models/toy_normal_final.zip",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/perfect/toy_normal/toy_normal_checkpoint_350000_steps.zip"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\nTesting with model: {model_path}")
            model = PPO.load(model_path)
            
            # Run one episode
            obs, _ = env.reset()
            done = False
            total_reward = 0
            scheduled_count = 0
            step_count = 0
            
            while not done and step_count < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                    scheduled_count += 1
            
            print(f"  Episode results:")
            print(f"    Jobs scheduled: {scheduled_count}/{env.total_tasks}")
            print(f"    Total reward: {total_reward:.2f}")
            print(f"    Steps taken: {step_count}")
            break
    
    # Analyze why jobs might not be scheduled
    print(f"\nPotential issues:")
    
    # Check if deadlines are too tight
    impossible_jobs = 0
    for job_id, job_info in jobs_by_deadline.items():
        if job_info['processing_hours'] > job_info['deadline_hours']:
            impossible_jobs += 1
    
    if impossible_jobs > 0:
        print(f"  - {impossible_jobs} jobs have impossible deadlines (processing > deadline)")
    
    # Check reward balance
    print(f"\nReward structure analysis:")
    print(f"  - Completion reward: +50.0")
    print(f"  - Action bonus: +5.0")
    print(f"  - Invalid action: -5.0")
    print(f"  - Late penalty: varies by hours late")
    
    # Calculate potential rewards
    best_case_reward = env.total_tasks * (50.0 + 5.0)  # All jobs on time
    print(f"  - Best case total reward: {best_case_reward:.0f}")
    
    # Worst case if all late by 24 hours
    late_penalty_per_job = 24 * 0.05  # Assuming 0.05 per hour
    worst_case_reward = env.total_tasks * (50.0 + 5.0 - late_penalty_per_job)
    print(f"  - Worst case (all 24h late): {worst_case_reward:.0f}")
    
    print("\nRecommendations:")
    print("1. Increase completion reward to 100+ to outweigh late penalties")
    print("2. Use graduated late penalties instead of linear")
    print("3. Add intermediate rewards for partial progress")
    print("4. Consider relaxing deadlines for training")


if __name__ == "__main__":
    analyze_toy_normal()