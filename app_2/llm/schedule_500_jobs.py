"""
Schedule 500 production jobs and create job-centric Gantt chart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import json
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_scheduler import LLMScheduler
from llm.parser import ScheduleParser, ScheduledTask


def create_production_gantt(scheduled_tasks, title="Production Schedule - 500 Jobs"):
    """Create production Gantt chart with jobs on Y-axis."""
    
    # Create a taller figure to accommodate 500 jobs
    fig, ax = plt.subplots(figsize=(24, 80))  # Very tall for 500 jobs
    
    # Sort tasks by family and sequence
    scheduled_tasks.sort(key=lambda t: (t.family_id, t.sequence))
    
    # Group by family
    families = {}
    for task in scheduled_tasks:
        if task.family_id not in families:
            families[task.family_id] = []
        families[task.family_id].append(task)
    
    # Create color palette
    family_list = list(families.keys())
    colors = sns.color_palette("husl", len(family_list))
    family_colors = {family: colors[i % len(colors)] for i, family in enumerate(family_list)}
    
    # Time bounds
    all_starts = [task.start_time for task in scheduled_tasks]
    all_ends = [task.end_time for task in scheduled_tasks]
    min_time = min(all_starts)
    max_time = max(all_ends)
    
    # Plot jobs
    y_pos = 0
    y_labels = []
    y_positions = []
    
    for family_id, family_tasks in families.items():
        # Sort by sequence
        family_tasks.sort(key=lambda t: t.sequence)
        
        for task in family_tasks:
            y_labels.append(task.job_id)
            y_positions.append(y_pos)
            
            # Calculate position
            start_hours = (task.start_time - min_time).total_seconds() / 3600
            duration_hours = (task.end_time - task.start_time).total_seconds() / 3600
            
            # Color and style
            color = family_colors[family_id]
            
            # Create rectangle
            rect = Rectangle(
                (start_hours, y_pos - 0.35),
                duration_hours,
                0.7,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add sequence text only for wider bars
            if duration_hours > 10:
                seq_text = f"Seq {task.sequence}/{task.total_sequences}"
                ax.text(
                    start_hours + duration_hours/2,
                    y_pos,
                    seq_text,
                    ha='center',
                    va='center',
                    fontsize=6,
                    weight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
                )
            
            # Duration at end
            if duration_hours > 0.5:
                ax.text(
                    start_hours + duration_hours + 0.5,
                    y_pos,
                    f"{duration_hours:.1f}h",
                    ha='left',
                    va='center',
                    fontsize=5,
                    style='italic'
                )
            
            y_pos += 1
    
    # Set Y-axis
    ax.set_ylim(-1, y_pos)
    ax.set_yticks(y_positions[::10])  # Show every 10th label to avoid crowding
    ax.set_yticklabels(y_labels[::10], fontsize=7)
    
    # Set X-axis
    total_hours = (max_time - min_time).total_seconds() / 3600
    ax.set_xlim(-5, total_hours + 5)
    
    # Date/time labels
    x_ticks = []
    x_labels = []
    current_time = min_time
    hour_offset = 0
    
    while hour_offset <= total_hours:
        x_ticks.append(hour_offset)
        date_str = current_time.strftime("%m/%d")
        time_str = current_time.strftime("%H:%M")
        x_labels.append(f"{date_str}\n{time_str}")
        current_time += timedelta(hours=24)
        hour_offset += 24
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)
    
    # Grid
    ax.grid(True, axis='x', which='major', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.2)
    
    # Labels
    ax.set_xlabel('Date and Time', fontsize=14, weight='bold')
    ax.set_ylabel('Job ID (sorted by sequence within family)', fontsize=14, weight='bold')
    ax.set_title(title, fontsize=18, weight='bold')
    
    # Stats
    stats_text = (
        f"Total Jobs: {len(scheduled_tasks)}\n"
        f"Families: {len(families)}\n"
        f"Duration: {total_hours:.1f} hours ({total_hours/24:.1f} days)"
    )
    
    ax.text(0.02, 0.99, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig, ax


def main():
    """Schedule 500 jobs and create visualization."""
    print("=== SCHEDULING 500 PRODUCTION JOBS ===\n")
    
    scheduler = LLMScheduler()
    
    # Update config to handle more jobs per prompt
    scheduler.max_jobs_per_prompt = 100
    
    print("Loading all jobs from production snapshot...")
    print("This will take several minutes due to the large number of jobs...")
    
    # Schedule ALL jobs (up to 500)
    result = scheduler.schedule(
        snapshot_path="/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json",
        strategy="direct",  # Direct strategy for speed
        max_jobs=500  # Get up to 500 jobs
    )
    
    print(f"\n✓ Scheduled {result['metrics']['total_jobs']} jobs")
    print(f"  - Families: {result['metrics']['families_scheduled']}")
    print(f"  - Makespan: {result['metrics']['makespan_hours']:.1f} hours")
    print(f"  - Response time: {result['llm_metadata']['response_time']:.1f}s")
    print(f"  - Cost: ${result['llm_metadata']['cost_estimate']['total_cost']:.4f}")
    
    # Parse scheduled tasks
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            job_id = task_data['task_id']
            parts = job_id.split('-')
            
            # Parse sequence
            if '/' in parts[-1]:
                seq_parts = parts[-1].split('/')
                seq_text = seq_parts[0]
                seq_num = ''
                for char in reversed(seq_text):
                    if char.isdigit():
                        seq_num = char + seq_num
                    else:
                        break
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
    
    print("\nCreating visualization (this may take a moment for 500 jobs)...")
    
    # For 500 jobs, we might want to create multiple charts or a very large one
    if len(scheduled_tasks) > 100:
        # Create a summary chart for first 100 jobs
        fig_summary, _ = create_production_gantt(
            scheduled_tasks[:100], 
            title=f"Production Schedule - First 100 of {len(scheduled_tasks)} Jobs"
        )
        summary_path = f"{output_dir}/production_500_summary.png"
        fig_summary.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig_summary)
        print(f"✓ Saved summary chart (first 100 jobs): {summary_path}")
    
    # Create full chart
    fig_full, _ = create_production_gantt(scheduled_tasks)
    full_path = f"{output_dir}/production_500_full.png"
    fig_full.savefig(full_path, dpi=100, bbox_inches='tight')  # Lower DPI for huge image
    plt.close(fig_full)
    print(f"✓ Saved full chart (all {len(scheduled_tasks)} jobs): {full_path}")
    
    # Print summary
    print(f"\n=== SCHEDULING SUMMARY ===")
    print(f"Total jobs scheduled: {len(scheduled_tasks)}")
    print(f"Total families: {len(set(t.family_id for t in scheduled_tasks))}")
    print(f"Time span: {result['metrics']['makespan_hours']:.1f} hours ({result['metrics']['makespan_hours']/24:.1f} days)")
    
    # Sample sequence verification
    print("\n=== SAMPLE SEQUENCES (First 3 families) ===")
    families = {}
    for task in scheduled_tasks:
        if task.family_id not in families:
            families[task.family_id] = []
        families[task.family_id].append(task)
    
    for i, (family_id, tasks) in enumerate(families.items()):
        if i >= 3:
            break
        tasks.sort(key=lambda t: t.sequence)
        print(f"\n{family_id}:")
        for task in tasks[:5]:  # Show up to 5 tasks per family
            print(f"  {task.sequence}/{task.total_sequences}: {task.job_id} ({task.processing_hours:.1f}h)")


if __name__ == "__main__":
    main()