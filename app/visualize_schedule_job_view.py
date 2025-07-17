#!/usr/bin/env python3
"""
Visualize production schedule with jobs on Y-axis.
Shows when each job is scheduled and on which machine.
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

def plot_schedule_job_view(env, max_steps=30):
    """Create Gantt chart with jobs on Y-axis."""
    
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
        
        # Get machine name
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
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, max(10, len(scheduled_jobs) * 0.3)))
    
    # Define colors for different machines
    machine_colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors
    machine_color_map = {}
    
    # Assign colors to machines
    unique_machines = list(set(job['machine_idx'] for job in scheduled_jobs))
    for idx, machine_idx in enumerate(unique_machines):
        machine_color_map[machine_idx] = machine_colors[idx % len(machine_colors)]
    
    # Plot settings
    time_horizon = 72  # Show 3 days (72 hours)
    
    # Draw break times as vertical bands
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
                
                # Handle breaks that span midnight
                if end_hour < start_hour:
                    # Draw from start to midnight
                    ax.axvspan(
                        day_offset + start_hour, day_offset + 24,
                        facecolor='lightgray' if "Machine Off" not in break_period.name else 'darkgray',
                        alpha=0.3, zorder=0
                    )
                    # Draw from midnight to end (next day)
                    if day < 2:  # Don't draw beyond the chart
                        ax.axvspan(
                            day_offset + 24, day_offset + 24 + end_hour,
                            facecolor='lightgray' if "Machine Off" not in break_period.name else 'darkgray',
                            alpha=0.3, zorder=0
                        )
                else:
                    # Normal break within same day
                    ax.axvspan(
                        day_offset + start_hour, day_offset + end_hour,
                        facecolor='lightgray' if "Machine Off" not in break_period.name else 'darkgray',
                        alpha=0.3, zorder=0
                    )
    
    # Draw jobs
    print("Drawing scheduled jobs...")
    job_y_positions = {}
    current_y = 0
    
    for idx, job in enumerate(scheduled_jobs):
        # Assign Y position to job
        job_y_positions[job['job_id']] = current_y
        
        # Draw job bar
        rect = patches.Rectangle(
            (job['start'], current_y - 0.4), 
            job['end'] - job['start'], 0.8,
            facecolor=machine_color_map[job['machine_idx']],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
        # Add machine name on the bar
        ax.text(
            (job['start'] + job['end']) / 2, 
            current_y,
            f"{job['machine_name']}",
            ha='center', va='center', fontsize=8, weight='bold'
        )
        
        # Add LCD days on the right of the bar
        if job['is_important']:
            ax.text(
                job['end'] + 0.2, 
                current_y,
                f"({job['lcd_days']:.0f}d) ❗",
                ha='left', va='center', fontsize=8, color='red'
            )
        
        current_y += 1
    
    # Draw day boundaries
    for day in range(4):
        ax.axvline(x=day * 24, color='black', linestyle='--', alpha=0.5)
        if day < 3:
            day_date = env.base_date + timedelta(days=day)
            ax.text(
                day * 24 + 12, len(scheduled_jobs),
                day_date.strftime('%A\n%m/%d'),
                ha='center', va='bottom', fontsize=10, weight='bold'
            )
    
    # Set axis labels and limits
    ax.set_xlim(0, time_horizon)
    ax.set_ylim(-0.5, len(scheduled_jobs) - 0.5)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Jobs', fontsize=12)
    ax.set_title('Production Schedule - Job View', fontsize=14, weight='bold')
    
    # Set y-axis ticks with job names
    ax.set_yticks(range(len(scheduled_jobs)))
    job_labels = [job['job_id'] for job in scheduled_jobs]
    ax.set_yticklabels(job_labels, fontsize=9)
    
    # Set x-axis ticks for every 6 hours
    ax.set_xticks(range(0, time_horizon + 1, 6))
    ax.set_xticklabels([f'{h:02d}:00' for h in [h % 24 for h in range(0, time_horizon + 1, 6)]])
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.1)
    
    # Create machine legend
    legend_elements = []
    for machine_idx in sorted(machine_color_map.keys()):
        machine_name = env.machines[machine_idx]['machine_name']
        color = machine_color_map[machine_idx]
        legend_elements.append(patches.Patch(color=color, label=machine_name))
    
    # Add break legend items
    legend_elements.extend([
        patches.Patch(color='lightgray', alpha=0.3, label='Break Time'),
        patches.Patch(color='darkgray', alpha=0.3, label='Machine Off')
    ])
    
    # Position legend outside the plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              title='Machines & Breaks', fontsize=8)
    
    # Add statistics
    stats_text = f"Total Jobs: {len(scheduled_jobs)}\n"
    stats_text += f"Important Jobs: {sum(1 for j in scheduled_jobs if j['is_important'])}\n"
    stats_text += f"Machines Used: {len(unique_machines)}\n"
    stats_text += f"Time Span: {max(j['end'] for j in scheduled_jobs):.1f}h"
    
    ax.text(
        time_horizon - 1, -0.3,
        stats_text,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/carrickcheah/Project/ppo/app/visualizations/schedule_job_view.png'
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nJob view visualization saved to: {output_file}")
    
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
    plot_schedule_job_view(env, max_steps=25)

if __name__ == "__main__":
    main()