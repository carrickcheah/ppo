#!/usr/bin/env python3
"""
Visualize production schedule with machines on Y-axis.
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
from src.environments.break_time_constraints import BreakTimeConstraints

def plot_schedule_machine_view(env, max_steps=30):
    """Create Gantt chart with machines on Y-axis."""
    
    # Initialize environment
    obs, info = env.reset()
    
    # Track scheduled jobs
    scheduled_jobs = []
    
    # Take steps to schedule jobs
    print("Scheduling jobs...")
    for step in range(max_steps):
        if len(env.valid_actions) == 0:
            print(f"No more valid actions at step {step}")
            break
            
        # Take first valid action
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get machine info
        machine_idx = int(info['on_machine'].split(' ')[0])
        machine_name = env.machines[machine_idx]['machine_name']
        
        # Record job info
        job_info = {
            'job_id': info['scheduled_job'],
            'machine_idx': machine_idx,
            'machine_name': machine_name,
            'start': info['start_time'],
            'end': info['end_time'],
            'is_important': info['is_important'],
            'lcd_days': info['lcd_days_remaining']
        }
        scheduled_jobs.append(job_info)
        
        if terminated or truncated:
            break
    
    print(f"Scheduled {len(scheduled_jobs)} jobs")
    
    # Get unique machines used
    machines_used = sorted(list(set(job['machine_idx'] for job in scheduled_jobs)))
    machine_y_map = {machine_idx: idx for idx, machine_idx in enumerate(machines_used)}
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, max(8, len(machines_used) * 0.6)))
    
    # Define colors
    job_colors = {
        'urgent': '#ff6b6b',      # Red for urgent (LCD < 7 days)
        'important': '#ffa500',   # Orange for important
        'normal': '#4ecdc4',      # Teal for normal
    }
    
    # Plot settings
    time_horizon = 72  # Show 3 days (72 hours)
    
    # Draw break times as background
    print("\nDrawing break times...")
    break_constraints = env.break_constraints
    
    for day in range(3):  # Show 3 days
        day_offset = day * 24
        
        # Get day of week
        current_date = env.base_date + timedelta(days=day)
        day_of_week = (current_date.weekday() + 1) % 7  # Convert to Sunday=0 format
        
        # Draw breaks for this day
        for break_period in break_constraints.breaks:
            if break_period.applies_to_day(day_of_week):
                # Convert break times to hours
                start_hour = break_period.start_time.hour + break_period.start_time.minute / 60
                end_hour = break_period.end_time.hour + break_period.end_time.minute / 60
                
                # Determine break color
                if "Machine Off" in break_period.name:
                    color = 'darkgray'
                elif "Sunday" in break_period.name or "Saturday" in break_period.name:
                    color = 'lightcoral'
                else:
                    color = 'lightgray'
                
                # Handle breaks that span midnight
                if end_hour < start_hour:
                    # Draw from start to midnight
                    rect = patches.Rectangle(
                        (day_offset + start_hour, -0.5), 
                        24 - start_hour, len(machines_used),
                        facecolor=color, alpha=0.3, edgecolor='none', zorder=0
                    )
                    ax.add_patch(rect)
                    # Draw from midnight to end (next day)
                    if day < 2:  # Don't draw beyond the chart
                        rect = patches.Rectangle(
                            (day_offset + 24, -0.5), 
                            end_hour, len(machines_used),
                            facecolor=color, alpha=0.3, edgecolor='none', zorder=0
                        )
                        ax.add_patch(rect)
                else:
                    # Normal break within same day
                    rect = patches.Rectangle(
                        (day_offset + start_hour, -0.5), 
                        end_hour - start_hour, len(machines_used),
                        facecolor=color, alpha=0.3, edgecolor='none', zorder=0
                    )
                    ax.add_patch(rect)
    
    # Draw jobs
    print("Drawing scheduled jobs...")
    for job in scheduled_jobs:
        y_pos = machine_y_map[job['machine_idx']]
        
        # Determine job color based on urgency
        if job['lcd_days'] < 7:
            color = job_colors['urgent']
        elif job['is_important']:
            color = job_colors['important']
        else:
            color = job_colors['normal']
        
        # Draw job bar
        rect = patches.Rectangle(
            (job['start'], y_pos - 0.4), 
            job['end'] - job['start'], 0.8,
            facecolor=color, edgecolor='black', linewidth=1, zorder=1
        )
        ax.add_patch(rect)
        
        # Add job label
        job_label = job['job_id'].split('-')[-1]  # Show only last part for space
        ax.text(
            (job['start'] + job['end']) / 2, 
            y_pos,
            f"{job_label}\n({job['lcd_days']:.0f}d)",
            ha='center', va='center', fontsize=7, weight='bold'
        )
    
    # Draw machine utilization bars in background
    for machine_idx, y_pos in machine_y_map.items():
        # Calculate machine utilization
        machine_jobs = [j for j in scheduled_jobs if j['machine_idx'] == machine_idx]
        if machine_jobs:
            total_busy_time = sum(j['end'] - j['start'] for j in machine_jobs)
            utilization = total_busy_time / time_horizon * 100
            
            # Add utilization text
            ax.text(
                time_horizon + 0.5, y_pos,
                f"{utilization:.1f}%",
                ha='left', va='center', fontsize=9
            )
    
    # Draw day boundaries and labels
    for day in range(4):
        ax.axvline(x=day * 24, color='black', linestyle='--', alpha=0.5)
        if day < 3:
            day_date = env.base_date + timedelta(days=day)
            ax.text(
                day * 24 + 12, len(machines_used) - 0.3,
                day_date.strftime('%A\n%m/%d'),
                ha='center', va='bottom', fontsize=10, weight='bold'
            )
    
    # Draw time markers for shifts
    shift_times = [6.5, 23]  # 6:30 AM and 11:00 PM
    for day in range(3):
        for shift_time in shift_times:
            ax.axvline(x=day * 24 + shift_time, color='blue', linestyle=':', alpha=0.3)
    
    # Set axis labels and limits
    ax.set_xlim(0, time_horizon + 3)
    ax.set_ylim(-0.5, len(machines_used) - 0.5)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title('Production Schedule - Machine View', fontsize=14, weight='bold')
    
    # Set y-axis ticks with machine names
    ax.set_yticks(range(len(machines_used)))
    machine_labels = [env.machines[idx]['machine_name'] for idx in machines_used]
    ax.set_yticklabels(machine_labels, fontsize=10)
    
    # Set x-axis ticks for every 6 hours
    ax.set_xticks(range(0, time_horizon + 1, 6))
    ax.set_xticklabels([f'{h:02d}:00' for h in [h % 24 for h in range(0, time_horizon + 1, 6)]])
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.2)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=job_colors['urgent'], label='Urgent (< 7 days)'),
        patches.Patch(color=job_colors['important'], label='Important'),
        patches.Patch(color=job_colors['normal'], label='Normal'),
        patches.Patch(color='lightgray', alpha=0.3, label='Break Time'),
        patches.Patch(color='darkgray', alpha=0.3, label='Machine Off'),
        patches.Patch(color='lightcoral', alpha=0.3, label='Weekend')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add statistics box
    total_jobs = len(scheduled_jobs)
    urgent_jobs = sum(1 for j in scheduled_jobs if j['lcd_days'] < 7)
    important_jobs = sum(1 for j in scheduled_jobs if j['is_important'])
    avg_utilization = sum(
        sum(j['end'] - j['start'] for j in scheduled_jobs if j['machine_idx'] == m) / time_horizon * 100
        for m in machines_used
    ) / len(machines_used) if machines_used else 0
    
    stats_text = f"Statistics:\n"
    stats_text += f"Total Jobs: {total_jobs}\n"
    stats_text += f"Urgent Jobs: {urgent_jobs}\n"
    stats_text += f"Important Jobs: {important_jobs}\n"
    stats_text += f"Machines Used: {len(machines_used)}\n"
    stats_text += f"Avg Utilization: {avg_utilization:.1f}%"
    
    ax.text(
        0.5, -0.3,
        stats_text,
        ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        fontsize=9,
        transform=ax.transAxes
    )
    
    # Label for utilization column
    ax.text(
        time_horizon + 0.5, len(machines_used) - 0.3,
        "Util %",
        ha='left', va='bottom', fontsize=9, weight='bold'
    )
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/carrickcheah/Project/ppo/app/visualizations/schedule_machine_view.png'
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nMachine view visualization saved to: {output_file}")
    
    # plt.show()

def main():
    """Main function."""
    # Initialize environment
    env = ScaledProductionEnv(
        n_machines=10,
        max_episode_steps=100,
        max_valid_actions=20,
        data_file='app/data/large_production_data.json',
        snapshot_file='app/data/production_snapshot_latest.json'
    )
    
    # Create visualization
    plot_schedule_machine_view(env, max_steps=25)

if __name__ == "__main__":
    main()