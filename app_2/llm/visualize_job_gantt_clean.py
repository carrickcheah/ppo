"""
Create clean job-centric Gantt chart with proper sequence sorting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_scheduler import LLMScheduler
from llm.parser import ScheduleParser, ScheduledTask


def create_clean_job_gantt(scheduled_tasks, title="Production Schedule - Jobs by Sequence"):
    """Create clean Gantt chart with jobs properly sorted by sequence."""
    
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Sort tasks by family first, then by sequence number
    # This ensures 1/3 comes before 2/3 comes before 3/3
    scheduled_tasks.sort(key=lambda t: (t.family_id, t.sequence))
    
    # Group by family for better visualization
    families = {}
    for task in scheduled_tasks:
        if task.family_id not in families:
            families[task.family_id] = []
        families[task.family_id].append(task)
    
    # Create color palette
    family_list = list(families.keys())
    colors = sns.color_palette("husl", len(family_list))
    family_colors = {family: colors[i] for i, family in enumerate(family_list)}
    
    # Time bounds
    all_starts = [task.start_time for task in scheduled_tasks]
    all_ends = [task.end_time for task in scheduled_tasks]
    min_time = min(all_starts)
    max_time = max(all_ends)
    
    # Plot jobs
    y_pos = 0
    y_labels = []
    y_positions = []
    
    # Add spacing between families
    family_spacing = 0.5
    
    for family_id, family_tasks in families.items():
        # Sort family tasks by sequence
        family_tasks.sort(key=lambda t: t.sequence)
        
        for task in family_tasks:
            y_labels.append(task.job_id)
            y_positions.append(y_pos)
            
            # Calculate bar position
            start_hours = (task.start_time - min_time).total_seconds() / 3600
            duration_hours = (task.end_time - task.start_time).total_seconds() / 3600
            
            # Set color and style based on job type
            color = family_colors[family_id]
            edge_color = 'darkred' if len(task.machine_ids) > 10 else 'black'
            edge_width = 2.5 if len(task.machine_ids) > 10 else 1.5
            
            # Create rectangle
            rect = Rectangle(
                (start_hours, y_pos - 0.4),
                duration_hours,
                0.8,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            # Add text inside bar
            if duration_hours > 3:
                # Show sequence and machine count for multi-machine jobs
                if len(task.machine_ids) > 1:
                    bar_text = f"Seq {task.sequence}/{task.total_sequences} • {len(task.machine_ids)} machines"
                else:
                    bar_text = f"Seq {task.sequence}/{task.total_sequences}"
                
                ax.text(
                    start_hours + duration_hours/2,
                    y_pos,
                    bar_text,
                    ha='center',
                    va='center',
                    fontsize=9,
                    weight='bold',
                    color='white' if duration_hours > 10 else 'black',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='black' if duration_hours > 10 else 'white', 
                             alpha=0.7)
                )
            
            # Add duration at end
            ax.text(
                start_hours + duration_hours + 1,
                y_pos,
                f"{duration_hours:.1f}h",
                ha='left',
                va='center',
                fontsize=8,
                style='italic'
            )
            
            y_pos += 1
        
        # Add family separator
        if family_id != family_list[-1]:  # Not the last family
            ax.axhline(y=y_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
            y_pos += family_spacing
    
    # Set Y-axis
    ax.set_ylim(-1, y_pos)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Color code Y-labels by family
    for label, pos in zip(ax.get_yticklabels(), y_positions):
        job_id = label.get_text()
        for family_id, tasks in families.items():
            if any(t.job_id == job_id for t in tasks):
                label.set_color(family_colors[family_id])
                label.set_weight('bold')
                break
    
    # Set X-axis
    total_hours = (max_time - min_time).total_seconds() / 3600
    ax.set_xlim(-5, total_hours + 5)
    
    # Create date/time labels
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
    
    # Minor ticks every 6 hours
    minor_ticks = list(range(0, int(total_hours) + 6, 6))
    ax.set_xticks(minor_ticks, minor=True)
    
    # Grid
    ax.grid(True, axis='x', which='major', alpha=0.5, linewidth=1.5)
    ax.grid(True, axis='x', which='minor', alpha=0.3, linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.2)
    
    # Labels
    ax.set_xlabel('Date and Time', fontsize=14, weight='bold')
    ax.set_ylabel('Job ID (sorted by sequence within family)', fontsize=14, weight='bold')
    ax.set_title(title, fontsize=18, weight='bold', pad=20)
    
    # Add statistics box
    total_jobs = len(scheduled_tasks)
    multi_machine_jobs = [t for t in scheduled_tasks if len(t.machine_ids) > 1]
    large_multi_jobs = [t for t in scheduled_tasks if len(t.machine_ids) > 10]
    
    stats_text = (
        f"Total Jobs: {total_jobs}\n"
        f"Families: {len(families)}\n"
        f"Multi-Machine Jobs: {len(multi_machine_jobs)}\n"
        f"Large Multi-Machine (>10): {len(large_multi_jobs)}\n"
        f"Duration: {total_hours:.1f} hours ({total_hours/24:.1f} days)"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Add legend for multi-machine indication
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', 
                         linewidth=1.5, label='Single/Few Machine Jobs'),
        patches.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='darkred', 
                         linewidth=2.5, label='Large Multi-Machine Jobs (>10 machines)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig, ax


def main():
    """Create clean job-centric visualization."""
    print("=== CLEAN JOB GANTT CHART ===\n")
    
    scheduler = LLMScheduler()
    
    print("Scheduling real production jobs with proper sequencing...")
    
    # Try multiple times if we get too few jobs
    attempts = 0
    while attempts < 3:
        result = scheduler.schedule(
            snapshot_path="/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json",
            strategy="direct",  # Faster and more reliable
            max_jobs=30
        )
        
        if result['metrics']['total_jobs'] >= 15:
            break
        
        attempts += 1
        print(f"  Attempt {attempts}: Got {result['metrics']['total_jobs']} jobs, retrying...")
    
    print(f"✓ Scheduled {result['metrics']['total_jobs']} jobs")
    
    # Parse tasks
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            job_id = task_data['task_id']
            parts = job_id.split('-')
            
            # Parse sequence more carefully
            if '/' in parts[-1]:
                seq_parts = parts[-1].split('/')
                # Extract just the numeric part
                seq_text = seq_parts[0]
                # Find the sequence number
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
    
    fig, ax = create_clean_job_gantt(scheduled_tasks)
    output_path = f"{output_dir}/job_gantt_clean.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Saved clean job Gantt chart to: {output_path}")
    
    # Print sequence verification
    print("\n=== SEQUENCE VERIFICATION ===")
    families = {}
    for task in scheduled_tasks:
        if task.family_id not in families:
            families[task.family_id] = []
        families[task.family_id].append(task)
    
    for family_id, tasks in list(families.items())[:3]:  # Show first 3 families
        tasks.sort(key=lambda t: t.sequence)
        print(f"\nFamily {family_id}:")
        for task in tasks:
            print(f"  {task.sequence}/{task.total_sequences}: {task.job_id} "
                  f"({len(task.machine_ids)} machines, {task.processing_hours:.1f}h)")


if __name__ == "__main__":
    main()