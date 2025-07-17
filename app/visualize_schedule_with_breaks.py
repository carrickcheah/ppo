#!/usr/bin/env python3
"""
Visualize production schedule with break time constraints.
Shows how jobs are scheduled around breaks.
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
import json

def plot_schedule_with_breaks(env, max_steps=30):
    """Create Gantt chart showing schedule with break times."""
    
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
        
        # Record job info
        job_info = {
            'job_id': info['scheduled_job'],
            'machine': info['on_machine'].split(' ')[0],  # Extract machine index
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
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define colors
    job_color = 'skyblue'
    important_job_color = 'salmon'
    break_color = 'lightgray'
    machine_off_color = 'darkgray'
    
    # Plot settings
    n_machines = 10  # Show first 10 machines
    time_horizon = 72  # Show 3 days (72 hours)
    
    # Draw break times
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
                    rect = patches.Rectangle(
                        (day_offset + start_hour, -0.5), 
                        24 - start_hour, n_machines + 1,
                        facecolor=machine_off_color if "Machine Off" in break_period.name else break_color,
                        alpha=0.5,
                        edgecolor='none'
                    )
                    ax.add_patch(rect)
                    # Draw from midnight to end (next day)
                    if day < 2:  # Don't draw beyond the chart
                        rect = patches.Rectangle(
                            (day_offset + 24, -0.5), 
                            end_hour, n_machines + 1,
                            facecolor=machine_off_color if "Machine Off" in break_period.name else break_color,
                            alpha=0.5,
                            edgecolor='none'
                        )
                        ax.add_patch(rect)
                else:
                    # Normal break within same day
                    rect = patches.Rectangle(
                        (day_offset + start_hour, -0.5), 
                        end_hour - start_hour, n_machines + 1,
                        facecolor=machine_off_color if "Machine Off" in break_period.name else break_color,
                        alpha=0.5,
                        edgecolor='none'
                    )
                    ax.add_patch(rect)
                
                # Add break label
                if day == 0 and break_period.duration_minutes > 30:  # Only label longer breaks on first day
                    ax.text(
                        day_offset + (start_hour + end_hour) / 2, 
                        n_machines + 0.2,
                        break_period.name,
                        ha='center', va='bottom', fontsize=8, rotation=45
                    )
    
    # Draw jobs
    print("Drawing scheduled jobs...")
    for job in scheduled_jobs:
        machine_idx = int(job['machine'])
        if machine_idx < n_machines:  # Only show first N machines
            rect = patches.Rectangle(
                (job['start'], machine_idx - 0.4), 
                job['end'] - job['start'], 0.8,
                facecolor=important_job_color if job['is_important'] else job_color,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Add job label
            ax.text(
                (job['start'] + job['end']) / 2, 
                machine_idx,
                f"{job['job_id']}\n({job['lcd_days']:.0f}d)",
                ha='center', va='center', fontsize=8
            )
    
    # Draw day boundaries
    for day in range(4):
        ax.axvline(x=day * 24, color='black', linestyle='--', alpha=0.3)
        if day < 3:
            day_date = env.base_date + timedelta(days=day)
            ax.text(
                day * 24 + 12, n_machines + 0.8,
                day_date.strftime('%A'),
                ha='center', va='bottom', fontsize=10, weight='bold'
            )
    
    # Set axis labels and limits
    ax.set_xlim(0, time_horizon)
    ax.set_ylim(-0.5, n_machines + 1)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machine', fontsize=12)
    ax.set_title('Production Schedule with Break Time Constraints', fontsize=14, weight='bold')
    
    # Set y-axis ticks
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'M{i}' for i in range(n_machines)])
    
    # Set x-axis ticks for every 6 hours
    ax.set_xticks(range(0, time_horizon + 1, 6))
    ax.set_xticklabels([f'{h:02d}:00' for h in [h % 24 for h in range(0, time_horizon + 1, 6)]])
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=job_color, label='Regular Job'),
        patches.Patch(color=important_job_color, label='Important Job'),
        patches.Patch(color=break_color, alpha=0.5, label='Break Time'),
        patches.Patch(color=machine_off_color, alpha=0.5, label='Machine Off')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add statistics
    stats_text = f"Total Jobs: {len(scheduled_jobs)}\n"
    stats_text += f"Important Jobs: {sum(1 for j in scheduled_jobs if j['is_important'])}\n"
    stats_text += f"Machines Used: {len(set(int(j['machine']) for j in scheduled_jobs if int(j['machine']) < n_machines))}"
    ax.text(
        time_horizon - 1, -0.3,
        stats_text,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/Users/carrickcheah/Project/ppo/app/visualizations/schedule_with_breaks.png'
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSchedule visualization saved to: {output_file}")
    
    # plt.show()  # Comment out to avoid hanging

def main():
    """Main function."""
    # Initialize environment
    env = ScaledProductionEnv(
        n_machines=10,
        max_episode_steps=100,
        max_valid_actions=20,  # Reduced for faster execution
        data_file='app/data/large_production_data.json',
        snapshot_file='app/data/production_snapshot_latest.json'
    )
    
    # Create visualization
    plot_schedule_with_breaks(env, max_steps=20)  # Reduced steps

if __name__ == "__main__":
    main()