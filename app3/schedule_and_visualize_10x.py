#!/usr/bin/env python
"""
Schedule jobs with 10x model and generate visualization
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

def schedule_and_visualize_10x():
    """Schedule with 10x model and create visualization."""
    
    print("="*60)
    print("SCHEDULING WITH 10X MODEL & GENERATING VISUALIZATION")
    print("="*60)
    
    # Configuration
    data_path = 'data/100_jobs.json'  # Use 100 jobs for good visualization
    model_path = 'checkpoints/10x/best_model.pth'
    
    # Create environment
    print("\n1. Loading environment...")
    env = SchedulingEnv(data_path, max_steps=10000)
    print(f"   Total tasks: {len(env.loader.tasks)}")
    print(f"   Families: {len(env.loader.families)}")
    print(f"   Machines: {len(env.loader.machines)}")
    
    # Load 10x model
    print("\n2. Loading 10x model...")
    model = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=(512, 512, 256, 128),
        dropout_rate=0.1,
        use_batch_norm=False,
        exploration_rate=0,
        device='mps'
    )
    
    # Try to load weights
    if os.path.exists(model_path):
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='mps', weights_only=False)
            model_dict = model.policy.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['policy_state_dict'].items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.policy.load_state_dict(model_dict, strict=False)
            print("   Model loaded successfully")
        except Exception as e:
            print(f"   Warning: Using partially trained model - {e}")
    
    # Disable exploration
    if hasattr(model, 'set_training_mode'):
        model.set_training_mode(False)
    model.exploration_rate = 0
    
    # Run scheduling
    print("\n3. Scheduling jobs...")
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 10000:
        action, _ = model.predict(obs, info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        if steps % 100 == 0:
            print(f"   Progress: {info['tasks_scheduled']}/{info['total_tasks']} "
                  f"({info['tasks_scheduled']/info['total_tasks']*100:.1f}%)")
    
    print(f"\n   Final: {info['tasks_scheduled']}/{info['total_tasks']} tasks scheduled")
    
    # Collect scheduled tasks for visualization
    print("\n4. Preparing visualization data...")
    families = {}
    
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        family = env.loader.families[task.family_id]
        
        # Get color based on deadline
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
        
        # Extract sequence
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
    
    # Sort for ascending display (sequence 1 at top)
    print("\n5. Sorting tasks for display...")
    plot_tasks = []
    
    # Process families in reverse order for Y-axis display
    for family_id in sorted(families.keys(), reverse=True):
        family_tasks = families[family_id]
        
        # Sort sequences in descending order for bottom-up plotting
        family_tasks.sort(key=lambda x: x['sequence'], reverse=True)
        
        # Add to plot list
        plot_tasks.extend(family_tasks)
    
    # Select tasks to display (first 80 for readability)
    display_tasks = plot_tasks[:80]
    
    # Create visualization
    print("\n6. Creating Gantt chart...")
    fig, ax = plt.subplots(figsize=(22, 16))
    
    ax.set_title('10X MODEL SCHEDULE - 100 Jobs (Sequences in ASCENDING Order)', 
                 fontsize=20, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=14)
    ax.set_ylabel('Tasks (Family_Process)', fontsize=14)
    
    # Plot tasks
    y_labels = []
    for i, task in enumerate(display_tasks):
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
                str(task['sequence']),
                ha='center', va='center',
                fontsize=9,
                fontweight='bold',
                color='white' if task['color'] in ['#FF0000', '#00FF00'] else 'black'
            )
    
    # Format axes
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    
    # Invert Y-axis to show sequence 1 at top
    ax.invert_yaxis()
    
    if display_tasks:
        max_time = max(t['end'] for t in display_tasks)
        ax.set_xlim(0, max_time * 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.grid(True, alpha=0.1, axis='y')
    
    # Add deadline line
    ax.axvline(x=16*24, color='red', linestyle='--', linewidth=2.5, 
               alpha=0.7, label='16-day deadline')
    
    # Add time markers
    for day in [1, 3, 7, 14, 21, 30]:
        ax.axvline(x=day*24, color='gray', linestyle=':', alpha=0.3)
        ax.text(day*24, ax.get_ylim()[0]-1, f'{day}d', 
                ha='center', fontsize=8, color='gray')
    
    # Legend
    legend_elements = [
        patches.Patch(color='#FF0000', label='Late', alpha=0.9),
        patches.Patch(color='#FFA500', label='Warning (<24h)', alpha=0.9),
        patches.Patch(color='#FFFF00', label='Caution (<72h)', alpha=0.9),
        patches.Patch(color='#00FF00', label='OK (>72h)', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Stats box
    # Calculate metrics
    late_jobs = sum(1 for t in plot_tasks if t['color'] == '#FF0000')
    on_time = len(plot_tasks) - late_jobs
    
    stats = (
        f'10X MODEL PERFORMANCE\n'
        f'Tasks Scheduled: {info["tasks_scheduled"]}/{info["total_tasks"]} '
        f'({info["tasks_scheduled"]/info["total_tasks"]*100:.1f}%)\n'
        f'On-Time: {on_time}/{len(plot_tasks)} ({on_time/len(plot_tasks)*100:.1f}%)\n'
        f'Families: {len(families)}\n'
        f'Model: 512→512→256→128 (1.1M params)'
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "10x_model_schedule.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n7. Visualization saved to: {output_file}")
    
    plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("SCHEDULING COMPLETE")
    print("="*60)
    print(f"✓ Completion Rate: {info['tasks_scheduled']/info['total_tasks']*100:.1f}%")
    print(f"✓ Tasks Scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    print(f"✓ Visualization: {output_file}")
    print("="*60)

if __name__ == "__main__":
    schedule_and_visualize_10x()