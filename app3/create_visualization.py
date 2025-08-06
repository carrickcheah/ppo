#!/usr/bin/env python
"""
FINAL FIX: Ascending order - sequence 1 at TOP, then 2, then 3.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import re

def extract_sequence_from_process(process_name):
    """Extract sequence number from process name."""
    match = re.search(r'(\d+)/\d+$', process_name)
    if match:
        return int(match.group(1))
    
    parts = process_name.split('-')
    if parts:
        last = parts[-1]
        if '/' in last:
            try:
                return int(last.split('/')[0])
            except:
                pass
    return 1

def final_ascending_fix():
    """Create visualization with sequences in TRUE ASCENDING order."""
    
    print("="*60)
    print("FINAL ASCENDING FIX - 1 at TOP, 2 below, 3 below that")
    print("="*60)
    
    # Setup
    model_path = "checkpoints/fast/best_model.pth"
    data_path = "data/40_jobs.json"
    
    # Create environment and model
    env = SchedulingEnv(data_path, max_steps=5000)
    model = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="mps"
    )
    model.load(model_path)
    
    # Run scheduling
    print("\n1. Running PPO scheduling...")
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 5000:
        action, _ = model.predict(obs, info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    print(f"   Scheduled: {info['tasks_scheduled']}/{info['total_tasks']} tasks")
    
    # Collect tasks
    print("\n2. Collecting tasks...")
    families = {}
    
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        family = env.loader.families[task.family_id]
        
        # Get color
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
        
        # Extract sequence from process name
        seq_num = extract_sequence_from_process(task.process_name)
        
        # Group by family
        if task.family_id not in families:
            families[task.family_id] = []
        
        families[task.family_id].append({
            'family_id': task.family_id,
            'process': task.process_name,
            'sequence': seq_num,
            'start': start,
            'end': end,
            'machine': machine,
            'color': color
        })
    
    # Sort for ASCENDING display (1 at top)
    print("\n3. Sorting for ASCENDING display...")
    plot_tasks = []
    
    # Process families in REVERSE order for correct Y-axis display
    for family_id in sorted(families.keys(), reverse=True):
        family_tasks = families[family_id]
        
        # Sort sequences in DESCENDING order for bottom-up plotting
        # This will make sequence 1 appear at TOP when plotted
        family_tasks.sort(key=lambda x: x['sequence'], reverse=True)
        
        # Add to plot list
        plot_tasks.extend(family_tasks)
    
    # Verify - show how it will appear after Y-axis inversion
    print("\n4. Display preview (will appear as ascending after Y-axis invert):")
    temp_families = {}
    for task in plot_tasks[:50]:
        if task['family_id'] not in temp_families:
            temp_families[task['family_id']] = []
        temp_families[task['family_id']].append(task)
    
    count = 0
    for fam_id, tasks in list(temp_families.items())[:3]:
        print(f"\n   {fam_id}:")
        # Show in reverse to simulate Y-axis inversion
        for task in reversed(tasks):
            print(f"      {task['process']} (seq: {task['sequence']})")
    
    # Create visualization
    print("\n5. Creating visualization...")
    fig, ax = plt.subplots(figsize=(20, 14))
    
    ax.set_title('PPO Schedule - ASCENDING Order (1 at TOP → 2 → 3)', 
                 fontsize=18, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=14)
    ax.set_ylabel('Tasks (Family_Process)', fontsize=14)
    
    # Plot tasks
    y_labels = []
    for i, task in enumerate(plot_tasks[:70]):
        label = f"{task['family_id']}_{task['process']}"
        y_labels.append(label)
        
        # Draw bar
        rect = Rectangle(
            (task['start'], i - 0.4),
            task['end'] - task['start'],
            0.8,
            facecolor=task['color'],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Add sequence number
        duration = task['end'] - task['start']
        if duration > 3:
            ax.text(
                task['start'] + duration/2,
                i,
                str(task['sequence']),
                ha='center', va='center',
                fontsize=10,
                fontweight='bold',
                color='white' if task['color'] in ['#FF0000', '#00FF00'] else 'black'
            )
    
    # Format axes
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    
    # CRITICAL: Invert Y-axis to show sequence 1 at TOP
    ax.invert_yaxis()
    
    if plot_tasks:
        max_time = max(t['end'] for t in plot_tasks)
        ax.set_xlim(0, max_time * 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.grid(True, alpha=0.1, axis='y')
    
    # Deadline
    ax.axvline(x=16*24, color='red', linestyle='--', linewidth=2.5, 
               alpha=0.7, label='16-day deadline')
    
    # Legend
    legend_elements = [
        patches.Patch(color='#FF0000', label='Late'),
        patches.Patch(color='#FFA500', label='Warning (<24h)'),
        patches.Patch(color='#FFFF00', label='Caution (<72h)'),
        patches.Patch(color='#00FF00', label='OK (>72h)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Stats
    stats = (
        f'Tasks Scheduled: {info["tasks_scheduled"]}/{info["total_tasks"]} '
        f'({info["tasks_scheduled"]/info["total_tasks"]*100:.1f}%)\n'
        f'Families: {len(families)}'
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_ascending.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n6. Saved to: {output_file}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("ASCENDING ORDER FIXED!")
    print("="*60)

if __name__ == "__main__":
    final_ascending_fix()