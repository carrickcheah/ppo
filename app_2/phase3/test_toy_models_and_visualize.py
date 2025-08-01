"""
Test toy stage models using Phase 4 models as a demonstration
Generates schedules and creates both job and machine allocation charts
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase4.environments.small_balanced_env import SmallBalancedEnvironment
from phase4.environments.small_complex_env import SmallComplexEnvironment
from phase4.environments.small_bottleneck_env import SmallBottleneckEnvironment

# Toy stage mappings to phase4 environments
STAGE_MAPPING = {
    'toy_easy': {
        'env_class': SmallBalancedEnvironment,
        'model_path': '/home/azureuser/ppo/app_2/phase4/results/balanced/checkpoints/balanced_iter6_1050960_steps.zip',
        'description': 'Learn sequence rules',
        'data_path': '/home/azureuser/ppo/app_2/phase4/data/small_balanced_data.json'
    },
    'toy_normal': {
        'env_class': SmallBalancedEnvironment,
        'model_path': '/home/azureuser/ppo/app_2/phase4/results/balanced/checkpoints/balanced_iter5_1000768_steps.zip',
        'description': 'Learn deadlines',
        'data_path': '/home/azureuser/ppo/app_2/phase4/data/small_balanced_data.json'
    },
    'toy_hard': {
        'env_class': SmallComplexEnvironment,
        'model_path': '/home/azureuser/ppo/app_2/phase4/results/complex/checkpoints/complex_iter6_2107040_steps.zip',
        'description': 'Learn priorities',
        'data_path': '/home/azureuser/ppo/app_2/phase4/data/small_complex_data.json'
    },
    'toy_multi': {
        'env_class': SmallBottleneckEnvironment,
        'model_path': '/home/azureuser/ppo/app_2/phase4/results/bottleneck/checkpoints/bottleneck_iter12_3004048_steps.zip',
        'description': 'Learn multi-machine',
        'data_path': '/home/azureuser/ppo/app_2/phase4/data/small_bottleneck_data.json'
    }
}

def create_job_view_gantt(schedule_data, stage_name, output_dir):
    """Create job-view Gantt chart showing job timelines"""
    schedule = schedule_data['schedule']
    if not schedule:
        print(f"No scheduled jobs for {stage_name}")
        return
    
    # Sort by job_id for consistent ordering
    schedule.sort(key=lambda x: x['job_id'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Set up job positions
    job_ids = [task['job_id'] for task in schedule]
    y_positions = {job_id: i for i, job_id in enumerate(job_ids)}
    
    # Color scheme based on deadline status
    colors = {
        'late': '#FF4444',      # Red
        'warning': '#FFA500',   # Orange
        'caution': '#9370DB',   # Purple
        'ok': '#32CD32'         # Green
    }
    
    # Current time line (red dashed)
    current_time = 16 * 24  # 16 days in hours
    
    # Plot each job
    for task in schedule:
        y_pos = y_positions[task['job_id']]
        start = task['start_time']
        duration = task['processing_time']
        end = start + duration
        
        # Determine color based on LCD
        lcd_hours = task.get('lcd_hours', current_time + 7*24)
        if end > lcd_hours:
            color = colors['late']
            status = 'Late (<0h)'
        elif end > lcd_hours - 24:
            color = colors['warning']
            status = 'Warning (<24h)'
        elif end > lcd_hours - 72:
            color = colors['caution']
            status = 'Caution (<72h)'
        else:
            color = colors['ok']
            status = 'OK (>72h)'
        
        # Draw rectangle
        rect = patches.Rectangle((start, y_pos - 0.4), duration, 0.8,
                               linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Draw current time line
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(current_time + 2, len(job_ids) - 1, 'Current Time', rotation=0, 
            color='red', fontsize=10, ha='left')
    
    # Set labels
    ax.set_yticks(range(len(job_ids)))
    ax.set_yticklabels(job_ids, fontsize=10)
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('Jobs', fontsize=12)
    
    # Add title with stage info
    title = f'Job Allocation - {stage_name.upper()} ({STAGE_MAPPING[stage_name]["description"]})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set x-axis limits
    max_time = max(task['start_time'] + task['processing_time'] for task in schedule) + 48
    ax.set_xlim(0, max(24*24, max_time))
    ax.set_ylim(-0.5, len(job_ids) - 0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['late'], label='Late (<0h)'),
        patches.Patch(color=colors['warning'], label='Warning (<24h)'),
        patches.Patch(color=colors['caution'], label='Caution (<72h)'),
        patches.Patch(color=colors['ok'], label='OK (>72h)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Invert y-axis to match the standard style
    ax.invert_yaxis()
    
    # Save
    output_path = os.path.join(output_dir, f"{stage_name}_job_allocation.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Job allocation chart saved to: {output_path}")

def create_machine_view_gantt(schedule_data, stage_name, output_dir):
    """Create machine-view Gantt chart showing machine utilization"""
    schedule = schedule_data['schedule']
    machines = schedule_data['machines']
    
    if not schedule:
        print(f"No scheduled jobs for {stage_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Set up machine positions
    machine_names = [m['machine_name'] for m in machines]
    y_positions = {m['machine_id']: i for i, m in enumerate(machines)}
    
    # Color scheme
    colors = {
        'late': '#FF4444',      # Red
        'warning': '#FFA500',   # Orange
        'caution': '#9370DB',   # Purple
        'ok': '#32CD32'         # Green
    }
    
    # Current time line
    current_time = 16 * 24  # 16 days in hours
    
    # Group schedule by machine
    machine_schedules = {}
    for task in schedule:
        machine_id = task['machine_id']
        if machine_id not in machine_schedules:
            machine_schedules[machine_id] = []
        machine_schedules[machine_id].append(task)
    
    # Plot each task on machines
    for machine_id, tasks in machine_schedules.items():
        if machine_id not in y_positions:
            continue
            
        y_pos = y_positions[machine_id]
        
        for task in tasks:
            start = task['start_time']
            duration = task['processing_time']
            end = start + duration
            
            # Determine color
            lcd_hours = task.get('lcd_hours', current_time + 7*24)
            if end > lcd_hours:
                color = colors['late']
            elif end > lcd_hours - 24:
                color = colors['warning']
            elif end > lcd_hours - 72:
                color = colors['caution']
            else:
                color = colors['ok']
            
            # Draw rectangle with job label
            rect = patches.Rectangle((start, y_pos - 0.4), duration, 0.8,
                                   linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            
            # Add job label on the rectangle
            ax.text(start + duration/2, y_pos, task['job_id'], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw current time line
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Calculate machine utilization
    utilizations = []
    total_time = max(24*24, max(task['start_time'] + task['processing_time'] for task in schedule) if schedule else 24*24)
    
    for i, (machine_id, machine_name) in enumerate(zip([m['machine_id'] for m in machines], machine_names)):
        if machine_id in machine_schedules:
            busy_time = sum(task['processing_time'] for task in machine_schedules[machine_id])
            utilization = (busy_time / total_time) * 100
        else:
            utilization = 0
        utilizations.append(utilization)
    
    # Set labels with utilization
    machine_labels = [f"{name} ({util:.0f}%)" for name, util in zip(machine_names, utilizations)]
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machine_labels, fontsize=10)
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    
    # Add title
    title = f'Machine Allocation - {stage_name.upper()} ({STAGE_MAPPING[stage_name]["description"]})'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set x-axis limits
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, len(machines) - 0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['late'], label='Late (<0h)'),
        patches.Patch(color=colors['warning'], label='Warning (<24h)'),
        patches.Patch(color=colors['caution'], label='Caution (<72h)'),
        patches.Patch(color=colors['ok'], label='OK (>72h)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add average utilization info
    avg_util = np.mean(utilizations)
    ax.text(0.02, 0.98, f'Average Utilization: {avg_util:.1f}%', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Invert y-axis
    ax.invert_yaxis()
    
    # Save
    output_path = os.path.join(output_dir, f"{stage_name}_machine_allocation.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Machine allocation chart saved to: {output_path}")

def test_and_save_schedule(stage_name):
    """Test a model and save its schedule for visualization"""
    print(f"\nTesting {stage_name}:")
    print("-" * 50)
    
    stage_config = STAGE_MAPPING[stage_name]
    
    # Check if model exists
    if not os.path.exists(stage_config['model_path']):
        print(f"Model not found at {stage_config['model_path']}")
        return None
    
    print(f"Loading model from: {stage_config['model_path']}")
    model = PPO.load(stage_config['model_path'])
    
    # Create environment
    env = stage_config['env_class']()
    
    # Run one episode to generate schedule
    obs, _ = env.reset()
    done = False
    step_count = 0
    max_steps = 500
    episode_reward = 0
    
    schedule = []
    
    while not done and step_count < max_steps:
        # Use trained PPO model to predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        episode_reward += reward
        step_count += 1
        
        # Check if a job was scheduled
        if info.get('job_scheduled', False):
            job_info = info.get('scheduled_job_info', {})
            if job_info:
                schedule.append({
                    'job_id': job_info.get('job_id', f'Job_{len(schedule)}'),
                    'machine_id': job_info.get('machine_id', 0),
                    'machine_name': f"Machine {job_info.get('machine_id', 0)}",
                    'start_time': job_info.get('start_time', 0),
                    'processing_time': job_info.get('processing_time', 1),
                    'end_time': job_info.get('end_time', job_info.get('start_time', 0) + job_info.get('processing_time', 1)),
                    'lcd_hours': job_info.get('lcd_hours', 7 * 24),
                    'is_important': job_info.get('is_important', False)
                })
    
    # Get final metrics
    total_jobs = info.get('total_jobs', len(env.jobs) if hasattr(env, 'jobs') else 10)
    scheduled_jobs = len(schedule)
    completion_rate = scheduled_jobs / total_jobs if total_jobs > 0 else 0
    
    print(f"PPO scheduled {scheduled_jobs}/{total_jobs} jobs ({completion_rate:.1%})")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Steps taken: {step_count}")
    
    # Get machine info
    n_machines = env.n_machines if hasattr(env, 'n_machines') else 3
    machines = []
    for i in range(n_machines):
        machines.append({
            'machine_id': i,
            'machine_name': f'Machine {i}'
        })
    
    return {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'model_path': stage_config['model_path'],
        'metrics': {
            'total_jobs': total_jobs,
            'scheduled_jobs': scheduled_jobs,
            'completion_rate': completion_rate,
            'total_reward': episode_reward,
            'description': stage_config['description']
        },
        'schedule': schedule,
        'machines': machines
    }

def main():
    """Test all toy stages and create visualizations"""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Testing Toy Models and Generating Visualizations")
    print("=" * 60)
    print("Using trained PPO models to schedule jobs")
    print("=" * 60)
    
    # Create output directory
    output_dir = "/home/azureuser/ppo/app_2/visualizations/phase_3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save schedules directory
    schedules_dir = "/home/azureuser/ppo/app_2/phase3/ai_schedules"
    os.makedirs(schedules_dir, exist_ok=True)
    
    results = []
    
    for stage in stages:
        try:
            result = test_and_save_schedule(stage)
            if result:
                # Save schedule data
                schedule_path = os.path.join(schedules_dir, f"{stage}_schedule.json")
                with open(schedule_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved schedule to: {schedule_path}")
                
                # Create visualizations
                create_job_view_gantt(result, stage, output_dir)
                create_machine_view_gantt(result, stage, output_dir)
                
                results.append(result)
        except Exception as e:
            print(f"Error with {stage}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    print("\n" + "=" * 60)
    print("Summary of Results:")
    print("=" * 60)
    
    for result in results:
        stage = result['stage']
        metrics = result['metrics']
        print(f"\n{stage.upper()} ({metrics['description']}):")
        print(f"  - Scheduled: {metrics['scheduled_jobs']}/{metrics['total_jobs']} jobs")
        print(f"  - Completion Rate: {metrics['completion_rate']:.1%}")
        print(f"  - Total Reward: {metrics['total_reward']:.2f}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nVisualization files created:")
    for stage in stages:
        print(f"  - {stage}_job_allocation.png")
        print(f"  - {stage}_machine_allocation.png")

if __name__ == "__main__":
    main()