"""
Visualize LLM-generated schedules
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
from llm.parser import ScheduleParser


def create_gantt_chart(scheduled_tasks, title="LLM-Generated Schedule"):
    """Create a Gantt chart visualization of the schedule."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Color palette
    colors = sns.color_palette("husl", len(set(task.family_id for task in scheduled_tasks)))
    family_colors = {}
    for i, family_id in enumerate(set(task.family_id for task in scheduled_tasks)):
        family_colors[family_id] = colors[i]
    
    # Sort tasks by machine and start time
    tasks_by_machine = {}
    for task in scheduled_tasks:
        for machine_id in task.machine_ids:
            if machine_id not in tasks_by_machine:
                tasks_by_machine[machine_id] = []
            tasks_by_machine[machine_id].append(task)
    
    # Sort machines
    machine_ids = sorted(tasks_by_machine.keys())
    
    # Calculate time bounds
    all_starts = [task.start_time for task in scheduled_tasks]
    all_ends = [task.end_time for task in scheduled_tasks]
    min_time = min(all_starts)
    max_time = max(all_ends)
    
    # Plot tasks
    y_pos = 0
    y_labels = []
    
    for machine_id in machine_ids:
        y_labels.append(f"Machine {machine_id}")
        
        for task in sorted(tasks_by_machine[machine_id], key=lambda t: t.start_time):
            # Calculate position and width
            start_offset = (task.start_time - min_time).total_seconds() / 3600
            duration = (task.end_time - task.start_time).total_seconds() / 3600
            
            # Create rectangle
            rect = Rectangle(
                (start_offset, y_pos - 0.4),
                duration,
                0.8,
                facecolor=family_colors[task.family_id],
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Add task label
            label = f"{task.job_id.split('-')[1]}\n{task.sequence}/{task.total_sequences}"
            ax.text(
                start_offset + duration/2,
                y_pos,
                label,
                ha='center',
                va='center',
                fontsize=8,
                weight='bold'
            )
        
        y_pos += 1
    
    # Set axis properties
    ax.set_ylim(-0.5, len(machine_ids) - 0.5)
    ax.set_xlim(0, (max_time - min_time).total_seconds() / 3600)
    ax.set_yticks(range(len(machine_ids)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (hours from start)')
    ax.set_title(title, fontsize=16, weight='bold')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend for families
    legend_patches = []
    for family_id, color in family_colors.items():
        legend_patches.append(patches.Patch(color=color, label=family_id))
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add time markers
    time_marks = range(0, int((max_time - min_time).total_seconds() / 3600) + 1, 24)
    ax.set_xticks(time_marks)
    ax.set_xticklabels([f"{t}h\n(Day {t//24})" for t in time_marks])
    
    plt.tight_layout()
    return fig, ax


def create_performance_chart(metrics):
    """Create performance metrics visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Response Time vs Strategy
    strategies = ['direct', 'chain_of_thought', 'constraint_focused']
    response_times = [30, 51.16, 45]  # Example times
    ax1.bar(strategies, response_times, color='skyblue')
    ax1.set_title('Response Time by Strategy')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Strategy')
    
    # Token Usage Breakdown
    labels = ['Input Tokens', 'Output Tokens']
    sizes = [2081, 1274]
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
    ax2.set_title('Token Usage Distribution')
    
    # Cost Analysis
    job_counts = [5, 10, 50, 100]
    estimated_costs = [0.0006, 0.0012, 0.006, 0.012]
    ax3.plot(job_counts, estimated_costs, marker='o', linewidth=2, markersize=8)
    ax3.set_title('Estimated Cost vs Job Count')
    ax3.set_xlabel('Number of Jobs')
    ax3.set_ylabel('Cost ($)')
    ax3.grid(True, alpha=0.3)
    
    # Makespan Comparison
    methods = ['LLM\n(chain)', 'LLM\n(direct)', 'FIFO\n(baseline)', 'PPO\n(trained)']
    makespans = [170.6, 180.2, 195.5, 165.3]  # Example values
    colors = ['green' if m < 180 else 'orange' for m in makespans]
    ax4.bar(methods, makespans, color=colors)
    ax4.set_title('Makespan Comparison')
    ax4.set_ylabel('Makespan (hours)')
    ax4.axhline(y=180, color='red', linestyle='--', label='Target')
    ax4.legend()
    
    plt.tight_layout()
    return fig


def main():
    """Generate and visualize a schedule."""
    print("Generating LLM schedule visualization...")
    
    # Generate a small schedule
    scheduler = LLMScheduler()
    result = scheduler.schedule(max_jobs=8, strategy="chain_of_thought")
    
    # Parse the schedule
    parser = ScheduleParser()
    
    # Extract scheduled tasks from result
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            # Parse the task data to get ScheduledTask objects
            job_id = task_data['task_id']
            parts = job_id.split('-')
            
            # Extract sequence info
            if '/' in parts[-1]:
                seq_parts = parts[-1].split('/')
                sequence = int(seq_parts[0].split('-')[-1]) if '-' in seq_parts[0] else 1
                total_sequences = int(seq_parts[1])
            else:
                sequence = 1
                total_sequences = 1
            
            # Create scheduled task
            from llm.parser import ScheduledTask
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
    
    # Create visualizations
    os.makedirs('/Users/carrickcheah/Project/ppo/app_2/llm/visualizations', exist_ok=True)
    
    # Gantt chart
    fig1, ax1 = create_gantt_chart(scheduled_tasks, 
                                   f"LLM Schedule - {len(scheduled_tasks)} Jobs")
    fig1.savefig('/Users/carrickcheah/Project/ppo/app_2/llm/visualizations/gantt_chart.png', 
                 dpi=300, bbox_inches='tight')
    print("✓ Gantt chart saved")
    
    # Performance metrics
    fig2 = create_performance_chart(result.get('llm_metadata', {}))
    fig2.savefig('/Users/carrickcheah/Project/ppo/app_2/llm/visualizations/performance_metrics.png', 
                 dpi=300, bbox_inches='tight')
    print("✓ Performance metrics saved")
    
    # Print summary
    print(f"\n=== Schedule Summary ===")
    print(f"Jobs scheduled: {result['metrics']['total_jobs']}")
    print(f"Families: {result['metrics']['families_scheduled']}")
    print(f"Makespan: {result['metrics']['makespan_hours']:.1f} hours")
    print(f"Response time: {result['llm_metadata']['response_time']:.1f}s")
    print(f"Cost: ${result['llm_metadata']['cost_estimate']['total_cost']:.4f}")
    
    print(f"\nVisualizations saved to: /Users/carrickcheah/Project/ppo/app_2/llm/visualizations/")


if __name__ == "__main__":
    main()