"""
Test toy stage models using actual PPO AI predictions
Shows how the trained models intelligently schedule jobs
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

def test_ai_scheduling(stage_name):
    """Test a trained PPO model and capture its scheduling decisions"""
    print(f"\nTesting {stage_name} AI Model:")
    print("-" * 50)
    
    # Load trained model
    model_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/{stage_name}/final_model.zip"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    model = PPO.load(model_path)
    print(f"Loaded trained PPO model from {model_path}")
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    # Run episode with AI making decisions
    obs, _ = env.reset()
    done = False
    step_count = 0
    max_steps = 200
    
    schedule = []
    action_history = []
    
    print("\nAI Scheduling Decisions:")
    
    while not done and step_count < max_steps:
        # Let AI predict the next action
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute AI's decision
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        step_count += 1
        
        # Record AI's decision
        action_history.append({
            'step': step_count,
            'action': int(action),
            'reward': float(reward),
            'valid': info.get('action_valid', False)
        })
        
        # If AI scheduled a job successfully
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            job_info = info.get('scheduled_job_info', {})
            job_id = info.get('scheduled_job', 'Unknown')
            
            # Extract family and sequence info
            if '_seq' in job_id:
                family_id = job_id.split('_seq')[0]
                seq_info = job_id.split('_seq')[1] if '_seq' in job_id else '1'
            else:
                family_id = job_id
                seq_info = '1'
            
            # Get job details from environment
            job_details = {
                'job_id': job_id,
                'family_id': family_id,
                'sequence': seq_info,
                'machine_id': job_info.get('machine', 0),
                'start_time': job_info.get('start_time', 0),
                'processing_time': job_info.get('processing_time', 0),
                'end_time': job_info.get('start_time', 0) + job_info.get('processing_time', 0),
                'step_scheduled': step_count
            }
            
            # Add deadline info if available
            if hasattr(env, 'families') and family_id in env.families:
                family_data = env.families[family_id]
                lcd_days = family_data.get('lcd_days_remaining', 7)
                job_details['lcd_hours'] = lcd_days * 24
                job_details['is_important'] = family_data.get('is_important', False)
            
            schedule.append(job_details)
            
            print(f"Step {step_count}: AI scheduled {job_id} on machine {job_details['machine_id']} "
                  f"at time {job_details['start_time']:.1f} (reward: {reward:.2f})")
    
    # Get final metrics
    total_jobs = env.total_tasks
    scheduled_jobs = len(env.scheduled_jobs)
    completion_rate = scheduled_jobs / total_jobs if total_jobs > 0 else 0
    
    print(f"\nAI Performance Summary:")
    print(f"  - Scheduled: {scheduled_jobs}/{total_jobs} jobs ({completion_rate:.1%})")
    print(f"  - Total steps: {step_count}")
    print(f"  - Final reward: {sum(a['reward'] for a in action_history):.2f}")
    
    # Analyze sequence compliance
    sequence_violations = check_sequence_violations(schedule)
    print(f"  - Sequence violations: {sequence_violations}")
    
    # Analyze deadline performance
    late_jobs = sum(1 for job in schedule if job.get('lcd_hours', float('inf')) < job['end_time'])
    print(f"  - Late jobs: {late_jobs}/{len(schedule)}")
    
    # Get machine info
    machines = []
    if hasattr(env, 'machines'):
        machines = [{'machine_id': m.get('machine_id', i), 'machine_name': m.get('machine_name', f'M{i}')} 
                   for i, m in enumerate(env.machines)]
    else:
        # Fallback
        n_machines = getattr(env, 'n_machines', 5)
        machines = [{'machine_id': i, 'machine_name': f'M{i}'} for i in range(n_machines)]
    
    return {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'metrics': {
            'total_jobs': total_jobs,
            'scheduled_jobs': scheduled_jobs,
            'completion_rate': completion_rate,
            'sequence_violations': sequence_violations,
            'late_jobs': late_jobs,
            'total_reward': sum(a['reward'] for a in action_history),
            'total_steps': step_count
        },
        'schedule': schedule,
        'machines': machines,
        'action_history': action_history[:50]  # First 50 actions for analysis
    }

def check_sequence_violations(schedule):
    """Check if AI respected sequence constraints"""
    violations = 0
    family_sequences = {}
    
    for job in schedule:
        family_id = job['family_id']
        if family_id not in family_sequences:
            family_sequences[family_id] = []
        
        # Parse sequence number
        seq_str = job['sequence']
        if '/' in seq_str:
            seq_num = int(seq_str.split('/')[0])
        else:
            seq_num = 1
        
        family_sequences[family_id].append({
            'sequence': seq_num,
            'start_time': job['start_time'],
            'job_id': job['job_id']
        })
    
    # Check each family
    for family_id, jobs in family_sequences.items():
        jobs.sort(key=lambda x: x['sequence'])
        
        for i in range(1, len(jobs)):
            if jobs[i]['start_time'] < jobs[i-1]['start_time']:
                violations += 1
                print(f"    Sequence violation: {jobs[i]['job_id']} started before {jobs[i-1]['job_id']}")
    
    return violations

def main():
    """Test all toy stages with AI scheduling"""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Testing Toy Stage AI Models")
    print("=" * 60)
    print("These are REAL AI scheduling decisions from trained PPO models")
    print("No hardcoded logic - pure learned behavior!")
    print("=" * 60)
    
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/ai_schedules"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for stage in stages:
        try:
            result = test_ai_scheduling(stage)
            if result:
                all_results[stage] = result
                
                # Save individual schedule
                output_path = os.path.join(output_dir, f"{stage}_ai_schedule.json")
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved AI schedule to {output_path}")
                
        except Exception as e:
            print(f"Error testing {stage}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("AI SCHEDULING SUMMARY")
    print("=" * 60)
    
    for stage, result in all_results.items():
        metrics = result['metrics']
        print(f"\n{stage}:")
        print(f"  - AI scheduled: {metrics['scheduled_jobs']}/{metrics['total_jobs']} ({metrics['completion_rate']:.1%})")
        print(f"  - Sequence violations: {metrics['sequence_violations']}")
        print(f"  - Late jobs: {metrics['late_jobs']}")
        print(f"  - Total reward: {metrics['total_reward']:.2f}")
        
        # Show what AI learned
        if stage == 'toy_easy' and metrics['sequence_violations'] == 0:
            print("  ✓ AI learned sequence rules!")
        elif stage == 'toy_normal' and metrics['late_jobs'] < metrics['scheduled_jobs'] * 0.3:
            print("  ✓ AI learned to consider deadlines!")
        elif stage == 'toy_hard':
            important_jobs = sum(1 for j in result['schedule'] if j.get('is_important', False))
            if important_jobs > 0:
                print(f"  ✓ AI scheduled {important_jobs} important jobs!")
        elif stage == 'toy_multi':
            multi_machine_jobs = sum(1 for j in result['schedule'] if '-' in j.get('machine_id', ''))
            if multi_machine_jobs > 0:
                print(f"  ✓ AI handled {multi_machine_jobs} multi-machine jobs!")

if __name__ == "__main__":
    main()