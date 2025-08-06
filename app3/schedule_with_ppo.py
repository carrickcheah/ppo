#!/usr/bin/env python
"""
Use BEST PPO model to schedule jobs and visualize.
Model makes ALL decisions - no hardcoded logic!
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np

def schedule_and_visualize():
    """Use PPO model to schedule and create visualization."""
    
    print("="*60)
    print("PPO MODEL SCHEDULING - NO HARDCODED LOGIC")
    print("="*60)
    
    # Use BEST model
    model_path = "checkpoints/fast/best_model.pth"
    data_path = "data/40_jobs.json"
    
    # Create environment
    env = SchedulingEnv(data_path, max_steps=5000)
    print(f"\nEnvironment: {env.n_tasks} tasks, {env.n_machines} machines")
    
    # Load BEST model
    model = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="mps"
    )
    model.load(model_path)
    print(f"Loaded BEST model from {model_path}")
    
    # PPO MODEL SCHEDULES EVERYTHING
    print("\nPPO Model scheduling...")
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 5000:
        # PPO decides which task
        action, _ = model.predict(obs, info['action_mask'], deterministic=True)
        
        # Execute PPO's decision
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        if steps % 500 == 0:
            print(f"  Step {steps}: {info['tasks_scheduled']} tasks scheduled")
    
    print(f"\nScheduling complete!")
    print(f"Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']} ({info['tasks_scheduled']/info['total_tasks']*100:.1f}%)")
    print(f"Steps taken: {steps}")
    
    # Check sequence order
    print("\nChecking sequence constraints...")
    violations = 0
    family_tasks = {}
    
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        if task.family_id not in family_tasks:
            family_tasks[task.family_id] = []
        family_tasks[task.family_id].append({
            'sequence': task.sequence,
            'start': start,
            'end': end,
            'process': task.process_name,
            'machine': machine
        })
    
    # Check each family
    for family_id, tasks in family_tasks.items():
        tasks.sort(key=lambda x: x['sequence'])
        for i in range(len(tasks)-1):
            if tasks[i+1]['start'] < tasks[i]['end']:
                violations += 1
                print(f"  VIOLATION: {family_id} seq {tasks[i+1]['sequence']} starts before seq {tasks[i]['sequence']} ends")
    
    print(f"Sequence violations: {violations}")
    
    # CREATE VISUALIZATION
    print("\nCreating visualization...")
    
    # Prepare data for visualization
    scheduled_tasks = []
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        family = env.loader.families[task.family_id]
        
        # Color by deadline
        lcd_hours = family.lcd_days_remaining * 24
        days_before = (lcd_hours - end) / 24
        
        if days_before < 0:
            color = '#FF0000'  # Late
        elif days_before < 1:
            color = '#FFA500'  # Warning
        elif days_before < 3:
            color = '#FFFF00'  # Caution
        else:
            color = '#00FF00'  # OK
        
        scheduled_tasks.append({
            'family_id': task.family_id,
            'sequence': task.sequence,
            'process': task.process_name,
            'start': start,
            'end': end,
            'machine': machine,
            'color': color
        })
    
    # Group by family and sort by the ACTUAL sequence number in process name
    family_groups = {}
    for task in scheduled_tasks:
        if task['family_id'] not in family_groups:
            family_groups[task['family_id']] = []
        family_groups[task['family_id']].append(task)
    
    # Sort each family by extracting the real sequence from process name
    sorted_tasks = []
    for family_id in sorted(family_groups.keys()):
        family_tasks = family_groups[family_id]
        # Extract sequence number from process name (e.g., "CP08-324-2/4" -> 2)
        for task in family_tasks:
            # Split by '-' and get the last part, then split by '/' to get sequence
            parts = task['process'].split('-')
            if len(parts) > 0 and '/' in parts[-1]:
                seq_str = parts[-1].split('/')[0]
                try:
                    task['display_sequence'] = int(seq_str)
                except:
                    task['display_sequence'] = task['sequence']
            else:
                task['display_sequence'] = task['sequence']
        
        # Sort by the extracted sequence number
        family_tasks.sort(key=lambda x: x['display_sequence'])
        sorted_tasks.extend(family_tasks)
    
    scheduled_tasks = sorted_tasks
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 12))
    
    ax.set_title('PPO Model Schedule - Each Sequence on Own Row', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('Tasks (Family_Process_Sequence)', fontsize=12)
    
    # Plot each task on its own row
    y_labels = []
    for i, task in enumerate(scheduled_tasks[:60]):  # Show first 60
        # Create label
        label = f"{task['family_id']}_{task['process']}"
        y_labels.append(label)
        
        # Draw bar
        rect = Rectangle(
            (task['start'], i - 0.4),
            task['end'] - task['start'],
            0.8,
            facecolor=task['color'],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Add sequence number in bar
        duration = task['end'] - task['start']
        if duration > 5:
            ax.text(
                task['start'] + duration/2,
                i,
                f"Seq {task['sequence']}",
                ha='center', va='center',
                fontsize=8,
                fontweight='bold',
                color='white' if task['color'] in ['#FF0000', '#00FF00'] else 'black'
            )
    
    # Format axes
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    
    # Set x limit
    if scheduled_tasks:
        max_time = max(t['end'] for t in scheduled_tasks)
        ax.set_xlim(0, max_time * 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.grid(True, alpha=0.1, axis='y')
    
    # Add deadline line
    ax.axvline(x=16*24, color='red', linestyle='--', linewidth=2, alpha=0.7, label='16-day deadline')
    
    # Legend
    legend_elements = [
        patches.Patch(color='#FF0000', label='Late', alpha=0.9),
        patches.Patch(color='#FFA500', label='Warning', alpha=0.9),
        patches.Patch(color='#FFFF00', label='Caution', alpha=0.9),
        patches.Patch(color='#00FF00', label='OK', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add statistics box
    stats_text = (
        f'Tasks Scheduled: {info["tasks_scheduled"]}/{info["total_tasks"]} ({info["tasks_scheduled"]/info["total_tasks"]*100:.1f}%)\n'
        f'Sequence Violations: {violations}\n'
        f'Families Scheduled: {len(family_tasks)}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ppo_schedule.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("PPO MODEL SCHEDULING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    schedule_and_visualize()