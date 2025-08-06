#!/usr/bin/env python
"""
Create Gantt chart visualization of the scheduled jobs.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict

def run_and_visualize():
    """Run scheduling and create visualization."""
    
    print("=" * 60)
    print("PPO SCHEDULER - GANTT CHART VISUALIZATION")
    print("=" * 60)
    
    # Setup
    model_path = "checkpoints/fast/model_40jobs.pth"
    data_path = "data/40_jobs.json"
    
    # Create environment
    env = SchedulingEnv(data_path, max_steps=1500)
    print(f"\nEnvironment: {env.n_tasks} tasks, {env.n_machines} machines")
    
    # Load model
    ppo = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="mps"
    )
    ppo.load(model_path)
    print("Model loaded successfully!")
    
    # Run scheduling
    print("\nRunning scheduling...")
    obs, info = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 1500:
        action, _ = ppo.predict(obs, info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
        if steps % 200 == 0:
            print(f"  Progress: {info['tasks_scheduled']}/{info['total_tasks']} tasks scheduled")
    
    # Results
    print(f"\nScheduling Complete!")
    print(f"  Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    print(f"  Completion rate: {info['tasks_scheduled']/info['total_tasks']*100:.1f}%")
    print(f"  Total reward: {total_reward:.1f}")
    
    # Create visualization
    create_gantt_chart(env, info)

def create_gantt_chart(env, info):
    """Create professional Gantt chart."""
    
    print("\nCreating Gantt chart...")
    
    # Collect schedule data
    schedule_data = []
    
    # Go through task_schedules to get the actual scheduled tasks
    for task_idx in env.task_schedules:
        task = env.loader.tasks[int(task_idx)]
        schedule_info = env.task_schedules[task_idx]
        # schedule_info is a tuple: (start_time, end_time, machine)
        start_time, end_time, machine = schedule_info
        
        schedule_data.append({
            'task_idx': task_idx,
            'family_id': task.family_id,
            'sequence': task.sequence,
            'machine': machine,
            'start_time': start_time,
            'end_time': end_time,
            'processing_time': task.processing_time,
            'process_name': task.process_name
        })
    
    if not schedule_data:
        print("No tasks scheduled!")
        return
    
    # Sort by start time
    schedule_data.sort(key=lambda x: x['start_time'])
    
    # Group by machine and family
    machines_used = sorted(list(set([s['machine'] for s in schedule_data])))
    families = sorted(list(set([s['family_id'] for s in schedule_data])))
    
    # Color map for families
    colors = plt.cm.Set3(np.linspace(0, 1, len(families)))
    family_colors = {family: colors[i] for i, family in enumerate(families)}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # ========== CHART 1: Machine View ==========
    ax1.set_title('Machine Allocation Schedule', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Time (Hours)', fontsize=12)
    ax1.set_ylabel('Machines', fontsize=12)
    
    # Plot tasks on machines
    for task in schedule_data:
        machine_y = machines_used.index(task['machine'])
        
        # Draw rectangle
        rect = patches.Rectangle(
            (task['start_time'], machine_y - 0.35),
            task['end_time'] - task['start_time'],
            0.7,
            linewidth=1.5,
            edgecolor='darkblue',
            facecolor=family_colors[task['family_id']],
            alpha=0.8
        )
        ax1.add_patch(rect)
        
        # Add text label if space permits
        duration = task['end_time'] - task['start_time']
        if duration > 3:  # Only add text if box is wide enough
            label = f"{task['family_id'][:10]}\nSeq {task['sequence']}"
            ax1.text(
                task['start_time'] + duration/2,
                machine_y,
                label,
                ha='center', va='center',
                fontsize=7, fontweight='bold'
            )
    
    # Format machine axis
    ax1.set_yticks(range(len(machines_used)))
    ax1.set_yticklabels(machines_used, fontsize=9)
    ax1.set_ylim(-0.5, len(machines_used) - 0.5)
    
    # Format time axis
    max_time = max([t['end_time'] for t in schedule_data])
    ax1.set_xlim(0, max_time * 1.02)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========== CHART 2: Job Family View ==========
    ax2.set_title('Job Family Progress', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Time (Hours)', fontsize=12)
    ax2.set_ylabel('Job Families', fontsize=12)
    
    # Group tasks by family
    family_tasks = defaultdict(list)
    for task in schedule_data:
        family_tasks[task['family_id']].append(task)
    
    # Sort families by start time
    sorted_families = sorted(
        family_tasks.keys(),
        key=lambda f: min([t['start_time'] for t in family_tasks[f]])
    )
    
    # Plot family progress
    for family_idx, family_id in enumerate(sorted_families):
        tasks = sorted(family_tasks[family_id], key=lambda x: x['sequence'])
        
        for task in tasks:
            # Draw rectangle
            rect = patches.Rectangle(
                (task['start_time'], family_idx - 0.35),
                task['end_time'] - task['start_time'],
                0.7,
                linewidth=1.5,
                edgecolor='darkgreen',
                facecolor=family_colors[family_id],
                alpha=0.8
            )
            ax2.add_patch(rect)
            
            # Add sequence label
            duration = task['end_time'] - task['start_time']
            label = f"Seq {task['sequence']}"
            if duration > 2:
                label += f"\n{task['process_name'][:15]}"
            
            ax2.text(
                task['start_time'] + duration/2,
                family_idx,
                label,
                ha='center', va='center',
                fontsize=7
            )
    
    # Format family axis
    ax2.set_yticks(range(len(sorted_families)))
    ax2.set_yticklabels(sorted_families, fontsize=9)
    ax2.set_ylim(-0.5, len(sorted_families) - 0.5)
    
    # Format time axis
    ax2.set_xlim(0, max_time * 1.02)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add overall title with statistics
    makespan = max([t['end_time'] for t in schedule_data])
    utilization = calculate_utilization(schedule_data, machines_used, makespan)
    
    fig.suptitle(
        f'PPO Production Schedule - {info["tasks_scheduled"]}/{info["total_tasks"]} Tasks Scheduled ({info["tasks_scheduled"]/info["total_tasks"]*100:.1f}%)\n' +
        f'Makespan: {makespan:.1f} hours | Machine Utilization: {utilization:.1f}%',
        fontsize=18, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    os.makedirs("/Users/carrickcheah/Project/ppo/app3/visualizations", exist_ok=True)
    output_file = "/Users/carrickcheah/Project/ppo/app3/visualizations/gantt_chart.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Gantt chart saved to: {output_file}")
    
    # Close plot to avoid hanging
    plt.close()

def calculate_utilization(schedule_data, machines, makespan):
    """Calculate overall machine utilization."""
    total_busy_time = sum([t['end_time'] - t['start_time'] for t in schedule_data])
    total_available_time = len(machines) * makespan
    return (total_busy_time / total_available_time) * 100 if total_available_time > 0 else 0

if __name__ == "__main__":
    run_and_visualize()