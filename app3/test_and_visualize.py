#!/usr/bin/env python
"""
Test the trained model and create visualization of the schedule.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime, timedelta

def run_scheduling(model_path, data_path):
    """Run scheduling with trained model and return schedule."""
    
    print(f"\nRunning scheduling with trained model...")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print("-" * 50)
    
    # Create environment
    env = SchedulingEnv(data_path, max_steps=1500)
    print(f"Environment: {env.n_tasks} tasks, {env.n_machines} machines")
    
    # Load model
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="mps"
    )
    
    if os.path.exists(model_path):
        ppo.load(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Error: Model not found at {model_path}")
        return None, None
    
    # Run scheduling episode
    obs, info = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    print("\nScheduling in progress...")
    while not done and steps < 1500:
        # Get action from model
        action_mask = info['action_mask']
        action, _ = ppo.predict(obs, action_mask, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
        # Progress indicator
        if steps % 100 == 0:
            print(f"  Steps: {steps}, Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    
    # Results
    print(f"\n{'='*50}")
    print("SCHEDULING RESULTS:")
    print(f"{'='*50}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    print(f"Completion rate: {info['tasks_scheduled']/info['total_tasks']*100:.1f}%")
    print(f"Steps taken: {steps}")
    print(f"{'='*50}\n")
    
    return env, total_reward

def create_gantt_chart(env, save_path="visualizations/"):
    """Create Gantt chart visualization of the schedule."""
    
    print("Creating Gantt chart visualization...")
    
    # Create directory if needed
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data for visualization
    schedule_data = []
    machine_names = []
    
    for machine_id, tasks in env.machine_schedules.items():
        # machine_id is already the machine name string
        machine_name = machine_id
        if machine_name not in machine_names:
            machine_names.append(machine_name)
        
        for task in tasks:
            schedule_data.append({
                'machine': machine_name,
                'machine_idx': machine_names.index(machine_name),
                'family_id': task['family_id'],
                'sequence': task['sequence'],
                'start': task['start_time'],
                'duration': task['processing_time'],
                'task_idx': task['task_idx']
            })
    
    if not schedule_data:
        print("No scheduled tasks to visualize")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color map for families
    families = list(set([d['family_id'] for d in schedule_data]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
    family_colors = {family: colors[i] for i, family in enumerate(families)}
    
    # Plot 1: Machine view (which machines are busy when)
    ax1.set_title('Machine Allocation View', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Machines')
    
    # Only show machines that have tasks
    used_machines = list(set([d['machine'] for d in schedule_data]))
    used_machines.sort()
    
    for task in schedule_data:
        machine_y = used_machines.index(task['machine'])
        rect = patches.Rectangle(
            (task['start'], machine_y - 0.4),
            task['duration'], 0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=family_colors[task['family_id']],
            alpha=0.7
        )
        ax1.add_patch(rect)
        
        # Add task label
        if task['duration'] > 2:  # Only label if wide enough
            ax1.text(
                task['start'] + task['duration']/2,
                machine_y,
                f"{task['family_id'][:8]}\nSeq {task['sequence']}",
                ha='center', va='center', fontsize=6
            )
    
    ax1.set_yticks(range(len(used_machines)))
    ax1.set_yticklabels(used_machines, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max([t['start'] + t['duration'] for t in schedule_data]))
    
    # Plot 2: Job view (when each job family is processed)
    ax2.set_title('Job Family Progress View', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Job Families')
    
    # Group by family
    family_tasks = {}
    for task in schedule_data:
        if task['family_id'] not in family_tasks:
            family_tasks[task['family_id']] = []
        family_tasks[task['family_id']].append(task)
    
    # Sort families by earliest start time
    sorted_families = sorted(family_tasks.keys(), 
                           key=lambda f: min([t['start'] for t in family_tasks[f]]))
    
    for family_idx, family_id in enumerate(sorted_families):
        tasks = family_tasks[family_id]
        for task in tasks:
            rect = patches.Rectangle(
                (task['start'], family_idx - 0.4),
                task['duration'], 0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=family_colors[family_id],
                alpha=0.7
            )
            ax2.add_patch(rect)
            
            # Add sequence number
            ax2.text(
                task['start'] + task['duration']/2,
                family_idx,
                f"Seq {task['sequence']}",
                ha='center', va='center', fontsize=7
            )
    
    ax2.set_yticks(range(len(sorted_families)))
    ax2.set_yticklabels(sorted_families, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max([t['start'] + t['duration'] for t in schedule_data]))
    
    # Add statistics
    total_makespan = max([t['start'] + t['duration'] for t in schedule_data])
    tasks_scheduled = len(env.task_schedules)
    fig.suptitle(
        f'PPO Scheduling Results - {env.n_tasks} Tasks on {env.n_machines} Machines\n' +
        f'Completion: {tasks_scheduled}/{env.n_tasks} tasks ' +
        f'({tasks_scheduled/env.n_tasks*100:.1f}%) | ' +
        f'Makespan: {total_makespan:.1f} hours',
        fontsize=16, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(save_path, "schedule_gantt_chart.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Gantt chart saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    return output_file

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("PPO SCHEDULER - TEST AND VISUALIZATION")
    print("=" * 60)
    
    # Use the trained model
    model_path = "checkpoints/fast/model_40jobs.pth"
    data_path = "data/40_jobs.json"
    
    # Run scheduling
    env, reward = run_scheduling(model_path, data_path)
    
    if env and reward:
        # Create visualization
        output_file = create_gantt_chart(env)
        
        print("\nVisualization complete!")
        print(f"Check the Gantt chart at: {output_file}")
    else:
        print("Scheduling failed. Please check the model path.")