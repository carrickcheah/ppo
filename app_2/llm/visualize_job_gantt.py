"""
Create job-centric Gantt chart with jobs on Y-axis and time on X-axis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import json
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_scheduler import LLMScheduler
from llm.parser import ScheduleParser, ScheduledTask


def create_job_gantt_chart(scheduled_tasks, title="Real Production Schedule - Job View"):
    """Create Gantt chart with jobs on Y-axis and dates on X-axis."""
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Sort tasks by family and sequence (1/3 before 2/3 before 3/3)
    scheduled_tasks.sort(key=lambda t: (t.family_id, t.sequence))
    
    # Create job list for Y-axis
    job_ids = [task.job_id for task in scheduled_tasks]
    
    # Color by family
    families = list(set(task.family_id for task in scheduled_tasks))
    colors = sns.color_palette("husl", len(families))
    family_colors = {family: colors[i] for i, family in enumerate(families)}
    
    # Time bounds
    all_starts = [task.start_time for task in scheduled_tasks]
    all_ends = [task.end_time for task in scheduled_tasks]
    min_time = min(all_starts)
    max_time = max(all_ends)
    
    # Plot each job
    y_pos = 0
    y_labels = []
    y_positions = []
    
    for i, task in enumerate(scheduled_tasks):
        y_labels.append(task.job_id)
        y_positions.append(y_pos)
        
        # Calculate bar position
        start_hours = (task.start_time - min_time).total_seconds() / 3600
        duration_hours = (task.end_time - task.start_time).total_seconds() / 3600
        
        # Create rectangle
        color = family_colors[task.family_id]
        
        # Multi-machine jobs have thicker bars
        bar_height = 0.8 if len(task.machine_ids) > 1 else 0.6
        
        rect = Rectangle(
            (start_hours, y_pos - bar_height/2),
            duration_hours,
            bar_height,
            facecolor=color,
            edgecolor='black',
            linewidth=2 if len(task.machine_ids) > 1 else 1,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add text inside the bar showing job sequence and processing info
        if duration_hours > 5:  # Only show text if bar is wide enough
            # For multi-machine jobs, show machine count
            if len(task.machine_ids) > 1:
                bar_text = f"{task.sequence}/{task.total_sequences} • {len(task.machine_ids)}M"
            else:
                bar_text = f"{task.sequence}/{task.total_sequences}"
            
            ax.text(
                start_hours + duration_hours/2,
                y_pos,
                bar_text,
                ha='center',
                va='center',
                fontsize=8,
                weight='bold',
                color='white' if len(task.machine_ids) > 1 else 'black'
            )
        
        # Add processing time at the end of bar
        time_text = f"{duration_hours:.1f}h"
        ax.text(
            start_hours + duration_hours + 0.5,
            y_pos,
            time_text,
            ha='left',
            va='center',
            fontsize=8
        )
        
        y_pos += 1
    
    # Set Y-axis
    ax.set_ylim(-0.5, len(scheduled_tasks) - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # Set X-axis with dates
    total_hours = (max_time - min_time).total_seconds() / 3600
    ax.set_xlim(0, total_hours)
    
    # Create custom X-axis labels with dates and hours
    x_ticks = []
    x_labels = []
    
    # Add major ticks every 24 hours (days)
    current_time = min_time
    hour_offset = 0
    while hour_offset <= total_hours:
        x_ticks.append(hour_offset)
        date_str = current_time.strftime("%Y-%m-%d")
        x_labels.append(f"{date_str}\n{hour_offset:.0f}h")
        current_time += timedelta(days=1)
        hour_offset += 24
    
    # Add minor ticks every 6 hours
    minor_ticks = []
    for h in range(0, int(total_hours) + 6, 6):
        if h % 24 != 0:  # Skip major tick positions
            minor_ticks.append(h)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xticks(minor_ticks, minor=True)
    
    # Grid
    ax.grid(True, axis='x', which='major', alpha=0.5, linewidth=1.5)
    ax.grid(True, axis='x', which='minor', alpha=0.3, linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Date and Time (hours from start)', fontsize=12, weight='bold')
    ax.set_ylabel('Job ID', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    
    # Add legend for families
    legend_elements = []
    for family, color in sorted(family_colors.items()):
        legend_elements.append(patches.Patch(color=color, label=f'Family: {family}'))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add summary info
    multi_jobs = [t for t in scheduled_tasks if len(t.machine_ids) > 1]
    info_text = (f"Total Jobs: {len(scheduled_tasks)}\n"
                f"Multi-Machine Jobs: {len(multi_jobs)}\n"
                f"Families: {len(families)}\n"
                f"Total Duration: {total_hours:.1f} hours")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Highlight multi-machine jobs
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    
    # Add machine info on right side
    for i, task in enumerate(scheduled_tasks):
        if len(task.machine_ids) > 1:
            machine_list = f"M: {','.join(map(str, task.machine_ids[:5]))}"
            if len(task.machine_ids) > 5:
                machine_list += f"... ({len(task.machine_ids)} total)"
            ax2.text(1.01, i, machine_list, transform=ax2.get_yaxis_transform(), 
                    fontsize=7, va='center', color='red')
    
    plt.tight_layout()
    return fig, ax


def main():
    """Create job-centric visualization of real production schedule."""
    print("=== JOB-CENTRIC GANTT CHART ===\n")
    
    # Load the previously scheduled data or create new schedule
    scheduler = LLMScheduler()
    
    print("Scheduling real production jobs...")
    result = scheduler.schedule(
        snapshot_path="/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json",
        strategy="direct",
        max_jobs=20  # Get a manageable number for clear visualization
    )
    
    print(f"✓ Scheduled {result['metrics']['total_jobs']} jobs")
    
    # Parse scheduled tasks
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            job_id = task_data['task_id']
            parts = job_id.split('-')
            
            # Parse sequence
            if '/' in parts[-1]:
                seq_parts = parts[-1].split('/')
                seq_num = ''.join(c for c in seq_parts[0] if c.isdigit())
                sequence = int(seq_num) if seq_num else 1
                total_sequences = int(seq_parts[1])
            else:
                sequence = 1
                total_sequences = 1
            
            task = ScheduledTask(
                job_id=job_id,
                family_id=family_id,
                sequence=sequence,
                total_sequences=total_sequences,
                machine_ids=task_data['machine_ids'],
                start_time=datetime.strptime(task_data['start_time'], "%Y-%m-%d %H:%M"),
                end_time=datetime.strptime(task_data['end_time'], "%Y-%m-%d %H:%M"),
                processing_hours=(datetime.strptime(task_data['end_time'], "%Y-%m-%d %H:%M") - 
                                 datetime.strptime(task_data['start_time'], "%Y-%m-%d %H:%M")).total_seconds() / 3600
            )
            scheduled_tasks.append(task)
    
    # Create visualization
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/llm"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = create_job_gantt_chart(scheduled_tasks)
    output_path = f"{output_dir}/job_gantt_chart.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved job-centric Gantt chart to: {output_path}")
    
    # Print some job details
    print("\n=== SAMPLE JOB DETAILS ===")
    for task in scheduled_tasks[:5]:
        print(f"\n{task.job_id}:")
        print(f"  - Family: {task.family_id}")
        print(f"  - Sequence: {task.sequence}/{task.total_sequences}")
        print(f"  - Machines: {len(task.machine_ids)} machines {task.machine_ids[:5]}{'...' if len(task.machine_ids) > 5 else ''}")
        print(f"  - Duration: {task.processing_hours:.1f} hours")
        print(f"  - Schedule: {task.start_time.strftime('%Y-%m-%d %H:%M')} to {task.end_time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()