#!/usr/bin/env python3
"""
Clean production schedule visualization with jobs on Y-axis.
Matches the exact style requested by the user.
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

def plot_clean_job_schedule(env, max_steps=100):
    """Create clean Gantt chart exactly as requested."""
    
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
    
    # Sort jobs by family and sequence for proper display
    # Extract family and sequence from full name for sorting
    def get_sort_key(job):
        full_name = job['full_name']
        # Parse JOST25050205_CP08-046-2/3 -> family: JOST25050205_CP08-046, seq: 2
        parts = full_name.split('-')
        if len(parts) >= 3:
            family_part = '-'.join(parts[:-1])  # Everything except last part
            seq_part = parts[-1].split('/')[0]  # Extract sequence number
            return (family_part, int(seq_part))
        return (full_name, 0)
    
    scheduled_jobs_sorted = sorted(scheduled_jobs, key=get_sort_key)
    
    # Create figure - adjust size based on jobs and time
    fig_height = max(12, len(scheduled_jobs_sorted) * 0.3)  # 0.3 inch per job
    fig_width = min(36, max(24, time_horizon * 0.3))  # Scale with time
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Color scheme - single color with different shades for urgency
    base_color = '#1f77b4'  # Blue
    urgent_color = '#ff4444'  # Red for urgent
    
    # Time settings - adjust based on actual schedule
    if scheduled_jobs_sorted:
        max_time = max(job['end'] for job in scheduled_jobs_sorted)
        time_horizon = int(max_time + 24)  # Add 1 day buffer
    else:
        time_horizon = 72  # Default 3 days
    
    # Draw jobs
    for idx, job in enumerate(scheduled_jobs_sorted):
        y_pos = idx
        
        # Determine color based on urgency
        if job['lcd_days'] < 7:
            color = urgent_color
        else:
            color = base_color
            
        # Draw job bar
        rect = patches.Rectangle(
            (job['start'], y_pos - 0.4), 
            job['end'] - job['start'], 0.8,
            facecolor=color,
            edgecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add end marker (red line like in the image)
        ax.axvline(x=job['end'], ymin=(y_pos-0.4)/len(scheduled_jobs), 
                   ymax=(y_pos+0.4)/len(scheduled_jobs), 
                   color='red', linewidth=2)
    
    # Configure axes
    ax.set_xlim(0, time_horizon)
    ax.set_ylim(-0.5, len(scheduled_jobs_sorted) - 0.5)
    
    # Y-axis: Job names
    ax.set_yticks(range(len(scheduled_jobs_sorted)))
    ax.set_yticklabels([job['full_name'] for job in scheduled_jobs_sorted], 
                       fontsize=11, fontfamily='monospace')
    ax.set_ylabel('Jobs', fontsize=14, fontweight='bold')
    
    # X-axis: Time in HH:00 format
    ax.set_xticks(range(0, time_horizon + 1, 1))
    x_labels = [f'{h%24:02d}:00' for h in range(0, time_horizon + 1)]
    ax.set_xticklabels(x_labels, rotation=90, fontsize=9)
    ax.set_xlabel('Timeline (MYT)', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Title
    ax.set_title('Production Schedule', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = '/Users/carrickcheah/Project/ppo/app/visualizations/schedule_job_clean.png'
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nClean job schedule saved to: {output_file}")

def main():
    """Main function."""
    env = ScaledProductionEnv(
        n_machines=10,
        max_episode_steps=200,  # Increased
        max_valid_actions=100,   # Increased to show more jobs
        data_file='app/data/large_production_data.json',
        snapshot_file='app/data/production_snapshot_latest.json'
    )
    
    plot_clean_job_schedule(env, max_steps=100)  # Schedule all jobs

if __name__ == "__main__":
    main()