#!/usr/bin/env python3
"""
Clean production schedule visualization with machines on Y-axis.
Shows machine utilization and job assignments over time.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np
from src.environments.scaled_production_env import ScaledProductionEnv

def generate_realistic_job_names(env):
    """Generate realistic job names matching production patterns."""
    job_name_map = {}
    
    # Realistic patterns from production
    job_patterns = [
        ('JOAW', '25050074', 'CM17-002'),
        ('JOAW', '25050075', 'CM17-005'),
        ('JOAW', '25050294', 'CD11-029'),
        ('JOST', '25040190', 'CP08-496'),
        ('JOST', '25050061', 'CK02-004'),
        ('JOST', '25050249', 'CP08-046'),
        ('JOST', '25050250', 'CP08-087'),
        ('JOST', '25050271', 'CP08-517'),
        ('JOST', '25050298', 'CV02-059'),
        ('JOST', '25050299', 'CV04-008'),
    ]
    
    pattern_idx = 0
    for family_id, family_data in env.families_data.items():
        prefix, order_num, process = job_patterns[pattern_idx % len(job_patterns)]
        
        # Adjust order number slightly for variety
        order_suffix = int(family_id.split('-')[0][-2:])
        adjusted_order = f"{int(order_num[:-2])}{order_suffix:02d}"
        
        total_tasks = len(family_data['tasks'])
        for task in family_data['tasks']:
            seq = task['sequence']
            simple_id = f"{family_id}-{seq}"
            full_name = f"{prefix}{adjusted_order}_{process}-{seq}/{total_tasks}"
            job_name_map[simple_id] = full_name
        
        pattern_idx += 1
    
    return job_name_map

def plot_clean_machine_schedule(env, max_steps=100):
    """Create clean Gantt chart with machines on Y-axis."""
    
    # Get job name mapping
    job_name_map = generate_realistic_job_names(env)
    
    # Initialize and run scheduling
    obs, info = env.reset()
    scheduled_jobs = []
    
    print("Scheduling jobs...")
    for step in range(max_steps):
        if len(env.valid_actions) == 0:
            break
            
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        machine_idx = int(info['on_machine'].split(' ')[0])
        simple_job_id = info['scheduled_job']
        
        job_info = {
            'job_id': simple_job_id,
            'full_name': job_name_map.get(simple_job_id, simple_job_id),
            'machine_idx': machine_idx,
            'machine_name': env.machines[machine_idx]['machine_name'],
            'start': info['start_time'],
            'end': info['end_time'],
            'is_important': info['is_important'],
            'lcd_days': info['lcd_days_remaining']
        }
        scheduled_jobs.append(job_info)
        
        if terminated or truncated:
            break
    
    print(f"Scheduled {len(scheduled_jobs)} jobs")
    
    # Get unique machines used (sorted by index)
    machines_used = sorted(list(set(job['machine_idx'] for job in scheduled_jobs)))
    machine_y_map = {machine_idx: idx for idx, machine_idx in enumerate(machines_used)}
    
    # Time settings - adjust based on actual schedule
    if scheduled_jobs:
        max_time = max(job['end'] for job in scheduled_jobs)
        time_horizon = int(max_time + 24)  # Add 1 day buffer
    else:
        time_horizon = 72  # Default 3 days
    
    # Create figure - adjust size based on machines and time
    fig_height = max(10, len(machines_used) * 0.8)  # 0.8 inch per machine
    fig_width = min(36, max(24, time_horizon * 0.15))  # Scale with time but cap at 36
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Simple single color for all jobs
    job_color = '#1f77b4'  # Blue
    
    # Draw jobs on machines
    for job in scheduled_jobs:
        y_pos = machine_y_map[job['machine_idx']]
        
        # Use same color for all jobs
        color = job_color
        
        # Draw job bar
        rect = patches.Rectangle(
            (job['start'], y_pos - 0.4), 
            job['end'] - job['start'], 0.8,
            facecolor=color, 
            edgecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add job name on the bar (shortened for space)
        job_label = job['full_name'].split('_')[-1]  # Show process and sequence
        if (job['end'] - job['start']) > 2:  # Only show label if bar is wide enough
            ax.text(
                (job['start'] + job['end']) / 2, 
                y_pos,
                job_label,
                ha='center', va='center', 
                fontsize=8, color='white',
                weight='bold'
            )
        
        # Add end marker
        ax.axvline(x=job['end'], 
                   ymin=(y_pos-0.4)/len(machines_used), 
                   ymax=(y_pos+0.4)/len(machines_used), 
                   color='darkred', linewidth=1.5)
    
    # Draw break times as vertical bands (subtle)
    for day in range(int(time_horizon / 24) + 1):
        day_offset = day * 24
        
        # Machine off time (1:00 - 6:30)
        ax.axvspan(day_offset + 1, day_offset + 6.5, 
                   facecolor='lightgray', alpha=0.2, zorder=0)
        
        # Lunch break (12:45 - 13:30)
        ax.axvspan(day_offset + 12.75, day_offset + 13.5, 
                   facecolor='lightgray', alpha=0.1, zorder=0)
    
    # Calculate and display machine utilization
    for machine_idx, y_pos in machine_y_map.items():
        machine_jobs = [j for j in scheduled_jobs if j['machine_idx'] == machine_idx]
        if machine_jobs:
            total_busy_time = sum(j['end'] - j['start'] for j in machine_jobs)
            # Calculate utilization based on available working hours
            working_hours_per_day = 16.5  # 24 - 5.5 (machine off) - 2 (breaks)
            total_days = time_horizon / 24
            available_hours = working_hours_per_day * total_days
            utilization = min(100, total_busy_time / available_hours * 100)
            
            # Add utilization percentage on the right
            ax.text(
                time_horizon + 1, y_pos,
                f"{utilization:.1f}%",
                ha='left', va='center', fontsize=10, weight='bold'
            )
    
    # Configure axes
    ax.set_xlim(0, time_horizon + 5)
    ax.set_ylim(-0.5, len(machines_used) - 0.5)
    
    # Y-axis: Machine names
    ax.set_yticks(range(len(machines_used)))
    machine_labels = [env.machines[idx]['machine_name'] for idx in machines_used]
    ax.set_yticklabels(machine_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Machines', fontsize=14, fontweight='bold')
    
    # X-axis: Time in HH:00 format
    # Show every 6 hours for readability
    x_ticks = range(0, int(time_horizon) + 1, 6)
    ax.set_xticks(x_ticks)
    x_labels = [f'{h%24:02d}:00' for h in x_ticks]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Timeline (MYT)', fontsize=14, fontweight='bold')
    
    # Add day markers
    for day in range(int(time_horizon / 24) + 1):
        ax.axvline(x=day * 24, color='black', linestyle='-', alpha=0.3, linewidth=1)
        if day < int(time_horizon / 24):
            ax.text(day * 24 + 12, len(machines_used) - 0.2, 
                    f'Day {day+1}', ha='center', va='bottom', fontsize=10)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Title
    ax.set_title('Machine Schedule', fontsize=16, fontweight='bold', pad=20)
    
    # No legend needed - keep it clean
    
    # Add utilization label
    ax.text(time_horizon + 1, len(machines_used) - 0.2, 
            'Util%', ha='left', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    
    # Save
    output_file = '/Users/carrickcheah/Project/ppo/app/visualizations/schedule_machine_clean.png'
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nClean machine schedule saved to: {output_file}")

def main():
    """Main function."""
    env = ScaledProductionEnv(
        n_machines=10,
        max_episode_steps=200,
        max_valid_actions=100,
        data_file='app/data/large_production_data.json',
        snapshot_file='app/data/production_snapshot_latest.json'
    )
    
    plot_clean_machine_schedule(env, max_steps=100)

if __name__ == "__main__":
    main()