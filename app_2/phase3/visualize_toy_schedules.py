"""
Visualize toy stage schedules with job-view and machine-view Gantt charts
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np

def load_schedule(stage_name):
    """Load schedule data for a stage"""
    # First try AI schedule
    ai_schedule_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/ai_schedules/{stage_name}_ai_schedule.json"
    if os.path.exists(ai_schedule_path):
        print(f"Loading AI-generated schedule from {ai_schedule_path}")
        with open(ai_schedule_path, 'r') as f:
            return json.load(f)
    
    # Fallback to regular schedule
    schedule_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/schedules/{stage_name}_schedule.json"
    if os.path.exists(schedule_path):
        with open(schedule_path, 'r') as f:
            return json.load(f)
    
    print(f"Schedule not found for {stage_name}")
    return None

def create_job_view_gantt(schedule_data, stage_name):
    """Create job-view Gantt chart (like image 1 & 2)"""
    schedule = schedule_data['schedule']
    if not schedule:
        print(f"No scheduled jobs for {stage_name}")
        return
    
    # Sort by job_id for consistent ordering
    schedule.sort(key=lambda x: x['job_id'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Set up job positions
    job_ids = [task['job_id'] for task in schedule]
    y_positions = {job_id: i for i, job_id in enumerate(job_ids)}
    
    # Color scheme based on deadline status
    colors = {
        'late': '#FF4444',      # Red
        'warning': '#FFA500',   # Orange
        'caution': '#9370DB',   # Purple
        'ok': '#32CD32'         # Green
    }
    
    # Current time line (red dashed)
    current_time = 16 * 24  # 16 days in hours
    
    # Plot each job
    for task in schedule:
        y_pos = y_positions[task['job_id']]
        start = task['start_time']
        duration = task['processing_time']
        end = start + duration
        
        # Determine color based on LCD
        lcd_hours = task.get('lcd_date', current_time + 7*24)
        if end > lcd_hours:
            color = colors['late']
            status = 'Late (<0h)'
        elif end > lcd_hours - 24:
            color = colors['warning']
            status = 'Warning (<24h)'
        elif end > lcd_hours - 72:
            color = colors['caution']
            status = 'Caution (<72h)'
        else:
            color = colors['ok']
            status = 'OK (>72h)'
        
        # Draw rectangle
        rect = patches.Rectangle((start, y_pos - 0.4), duration, 0.8,
                               linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Draw current time line
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set labels
    ax.set_yticks(range(len(job_ids)))
    ax.set_yticklabels(job_ids, fontsize=10)
    ax.set_xlabel('Time (Hours)', fontsize=12)
    
    # Add AI indicator if using AI schedule
    title_suffix = " (AI-Generated)" if 'model_path' in schedule_data else ""
    ax.set_title(f'Production Planning System - {stage_name.upper()}{title_suffix}', fontsize=16, fontweight='bold')
    
    # Set x-axis limits
    ax.set_xlim(0, max(24*24, max(task['start_time'] + task['processing_time'] for task in schedule) + 48))
    ax.set_ylim(-0.5, len(job_ids) - 0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['late'], label='Late (<0h)'),
        patches.Patch(color=colors['warning'], label='Warning (<24h)'),
        patches.Patch(color=colors['caution'], label='Caution (<72h)'),
        patches.Patch(color=colors['ok'], label='OK (>72h)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Invert y-axis to match the image style
    ax.invert_yaxis()
    
    # Save
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stage_name}_job_view.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Job-view Gantt saved to {output_path}")

def create_machine_view_gantt(schedule_data, stage_name):
    """Create machine-view Gantt chart (like image 3)"""
    schedule = schedule_data['schedule']
    machines = schedule_data['machines']
    
    if not schedule:
        print(f"No scheduled jobs for {stage_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Set up machine positions
    machine_names = [m['machine_name'] for m in machines]
    y_positions = {m['machine_id']: i for i, m in enumerate(machines)}
    
    # Color scheme
    colors = {
        'late': '#FF4444',      # Red
        'warning': '#FFA500',   # Orange
        'caution': '#9370DB',   # Purple
        'ok': '#32CD32'         # Green
    }
    
    # Current time line
    current_time = 16 * 24  # 16 days in hours
    
    # Group schedule by machine
    machine_schedules = {}
    for task in schedule:
        for machine_id in task['assigned_machines']:
            if machine_id not in machine_schedules:
                machine_schedules[machine_id] = []
            machine_schedules[machine_id].append(task)
    
    # Plot each task on machines
    for machine_id, tasks in machine_schedules.items():
        if machine_id not in y_positions:
            continue
            
        y_pos = y_positions[machine_id]
        
        for task in tasks:
            start = task['start_time']
            duration = task['processing_time']
            end = start + duration
            
            # Determine color
            lcd_hours = task.get('lcd_date', current_time + 7*24)
            if end > lcd_hours:
                color = colors['late']
            elif end > lcd_hours - 24:
                color = colors['warning']
            elif end > lcd_hours - 72:
                color = colors['caution']
            else:
                color = colors['ok']
            
            # Draw rectangle with job label
            rect = patches.Rectangle((start, y_pos - 0.4), duration, 0.8,
                                   linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            
            # Add job label on the rectangle
            ax.text(start + duration/2, y_pos, task['job_id'], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw current time line
    ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Calculate machine utilization
    utilizations = []
    total_time = max(24*24, max(task['start_time'] + task['processing_time'] for task in schedule) if schedule else 24*24)
    
    for i, (machine_id, machine_name) in enumerate(zip([m['machine_id'] for m in machines], machine_names)):
        if machine_id in machine_schedules:
            busy_time = sum(task['processing_time'] for task in machine_schedules[machine_id])
            utilization = (busy_time / total_time) * 100
        else:
            utilization = 0
        utilizations.append(utilization)
    
    # Set labels with utilization
    machine_labels = [f"{name} ({util:.0f}%)" for name, util in zip(machine_names, utilizations)]
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machine_labels, fontsize=10)
    ax.set_xlabel('Time (Hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title(f'Machine Allocation - {stage_name.upper()}', fontsize=16, fontweight='bold')
    
    # Set x-axis limits
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, len(machines) - 0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['late'], label='Late (<0h)'),
        patches.Patch(color=colors['warning'], label='Warning (<24h)'),
        patches.Patch(color=colors['caution'], label='Caution (<72h)'),
        patches.Patch(color=colors['ok'], label='OK (>72h)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add time period buttons at top (visual only)
    button_y = 1.02
    time_periods = ['1d', '2d', '3d', '4d', '5d', '7d', '14d', '21d', '1m', '2m', '3m', 'all']
    for i, period in enumerate(time_periods):
        x_pos = 0.1 + i * 0.07
        if period == '3d':
            ax.text(x_pos, button_y, period, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', edgecolor='blue'),
                   fontsize=10, ha='center')
        else:
            ax.text(x_pos, button_y, period, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', edgecolor='gray'),
                   fontsize=10, ha='center')
    
    # Invert y-axis
    ax.invert_yaxis()
    
    # Save
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stage_name}_machine_view.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Machine-view Gantt saved to {output_path}")

def main():
    """Create visualizations for all toy stages"""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Creating Toy Stage Visualizations")
    print("=" * 60)
    
    for stage in stages:
        print(f"\nProcessing {stage}...")
        print("-" * 40)
        
        # Load schedule
        schedule_data = load_schedule(stage)
        if not schedule_data:
            continue
        
        # Create both views
        create_job_view_gantt(schedule_data, stage)
        create_machine_view_gantt(schedule_data, stage)
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")

if __name__ == "__main__":
    main()