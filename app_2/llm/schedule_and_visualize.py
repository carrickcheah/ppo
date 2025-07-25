"""
Schedule production jobs using LLM and create comprehensive visualizations
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
from llm.validator import ScheduleValidator


def create_gantt_chart(scheduled_tasks, title="LLM-Generated Production Schedule", save_path=None):
    """Create a Gantt chart visualization of the schedule."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Color palette - use more colors for many families
    unique_families = list(set(task.family_id for task in scheduled_tasks))
    colors = sns.color_palette("husl", len(unique_families))
    family_colors = {family: colors[i] for i, family in enumerate(unique_families)}
    
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
    machine_utilization = {}
    
    for machine_id in machine_ids:
        y_labels.append(f"M{machine_id}")
        total_busy_time = 0
        
        for task in sorted(tasks_by_machine[machine_id], key=lambda t: t.start_time):
            # Calculate position and width
            start_offset = (task.start_time - min_time).total_seconds() / 3600
            duration = (task.end_time - task.start_time).total_seconds() / 3600
            total_busy_time += duration
            
            # Create rectangle
            rect = Rectangle(
                (start_offset, y_pos - 0.4),
                duration,
                0.8,
                facecolor=family_colors[task.family_id],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add task label (only if space allows)
            if duration > 5:  # Only show label if task is longer than 5 hours
                label = f"{task.sequence}/{task.total_sequences}"
                ax.text(
                    start_offset + duration/2,
                    y_pos,
                    label,
                    ha='center',
                    va='center',
                    fontsize=6,
                    weight='bold'
                )
        
        # Calculate utilization
        total_time = (max_time - min_time).total_seconds() / 3600
        machine_utilization[machine_id] = (total_busy_time / total_time) * 100 if total_time > 0 else 0
        
        y_pos += 1
    
    # Set axis properties
    ax.set_ylim(-0.5, len(machine_ids) - 0.5)
    ax.set_xlim(0, (max_time - min_time).total_seconds() / 3600)
    ax.set_yticks(range(len(machine_ids)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_title(title, fontsize=16, weight='bold')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add time markers
    time_marks = range(0, int((max_time - min_time).total_seconds() / 3600) + 1, 24)
    ax.set_xticks(time_marks)
    ax.set_xticklabels([f"Day {t//24}" for t in time_marks])
    
    # Add utilization info
    avg_utilization = np.mean(list(machine_utilization.values()))
    ax.text(0.02, 0.98, f"Avg Utilization: {avg_utilization:.1f}%", 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax, machine_utilization


def create_metrics_dashboard(result, scheduled_tasks, violations, save_path=None):
    """Create a comprehensive metrics dashboard."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Schedule Overview (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = [
        ('Total Jobs', result['metrics']['total_jobs']),
        ('Families', result['metrics']['families_scheduled']),
        ('Makespan (days)', result['metrics']['makespan_hours'] / 24),
        ('Response Time (s)', result['llm_metadata']['response_time']),
        ('Cost ($)', result['llm_metadata']['cost_estimate']['total_cost'])
    ]
    
    y_pos = np.arange(len(metrics))
    values = [m[1] for m in metrics]
    colors_bar = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
    
    bars = ax1.barh(y_pos, values, color=colors_bar)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m[0] for m in metrics])
    ax1.set_title('Schedule Overview', fontsize=12, weight='bold')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        if i < 2:  # Integer values
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(value)}', va='center')
        else:  # Float values
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{value:.2f}', va='center')
    
    # 2. Machine Utilization Distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    tasks_by_machine = {}
    for task in scheduled_tasks:
        for machine_id in task.machine_ids:
            if machine_id not in tasks_by_machine:
                tasks_by_machine[machine_id] = []
            tasks_by_machine[machine_id].append(task)
    
    utilizations = []
    for machine_id, tasks in tasks_by_machine.items():
        total_busy = sum((t.end_time - t.start_time).total_seconds() / 3600 for t in tasks)
        total_time = result['metrics']['makespan_hours']
        utilization = (total_busy / total_time * 100) if total_time > 0 else 0
        utilizations.append(utilization)
    
    ax2.hist(utilizations, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Utilization (%)')
    ax2.set_ylabel('Number of Machines')
    ax2.set_title('Machine Utilization Distribution', fontsize=12, weight='bold')
    ax2.axvline(x=np.mean(utilizations), color='red', linestyle='--', 
                label=f'Avg: {np.mean(utilizations):.1f}%')
    ax2.legend()
    
    # 3. Job Completion Timeline (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    completion_times = [(task.end_time - min(t.start_time for t in scheduled_tasks)).total_seconds() / 3600 
                       for task in scheduled_tasks]
    completion_times.sort()
    
    ax3.plot(range(len(completion_times)), completion_times, 'b-', linewidth=2)
    ax3.fill_between(range(len(completion_times)), completion_times, alpha=0.3)
    ax3.set_xlabel('Job Number')
    ax3.set_ylabel('Completion Time (hours)')
    ax3.set_title('Job Completion Timeline', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Token Usage Breakdown (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    token_data = result['llm_metadata']['cost_estimate']
    labels = ['Input Tokens', 'Output Tokens']
    sizes = [token_data['input_tokens'], token_data['output_tokens']]
    colors_pie = ['lightcoral', 'lightskyblue']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f%%', 
                                       colors=colors_pie, startangle=90)
    ax4.set_title(f'Token Usage (Total: {token_data["total_tokens"]})', 
                  fontsize=12, weight='bold')
    
    # 5. Constraint Violations (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    if violations:
        violation_types = {}
        for v in violations:
            if v.type not in violation_types:
                violation_types[v.type] = 0
            violation_types[v.type] += 1
        
        types = list(violation_types.keys())
        counts = list(violation_types.values())
        
        ax5.bar(range(len(types)), counts, color='red', alpha=0.7)
        ax5.set_xticks(range(len(types)))
        ax5.set_xticklabels(types, rotation=45, ha='right')
        ax5.set_ylabel('Count')
        ax5.set_title('Constraint Violations', fontsize=12, weight='bold')
    else:
        ax5.text(0.5, 0.5, 'No Violations!', ha='center', va='center', 
                fontsize=16, color='green', weight='bold', transform=ax5.transAxes)
        ax5.set_title('Constraint Violations', fontsize=12, weight='bold')
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # 6. Jobs per Family (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    family_counts = {}
    for task in scheduled_tasks:
        if task.family_id not in family_counts:
            family_counts[task.family_id] = 0
        family_counts[task.family_id] += 1
    
    # Show top 10 families
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    families = [f[0][-8:] for f in sorted_families]  # Show last 8 chars of family ID
    counts = [f[1] for f in sorted_families]
    
    ax6.bar(range(len(families)), counts, color='lightgreen')
    ax6.set_xticks(range(len(families)))
    ax6.set_xticklabels(families, rotation=45, ha='right')
    ax6.set_ylabel('Number of Jobs')
    ax6.set_title('Top 10 Job Families', fontsize=12, weight='bold')
    
    # 7. Processing Time Distribution (bottom)
    ax7 = fig.add_subplot(gs[2, :])
    processing_times = [task.processing_hours for task in scheduled_tasks]
    
    ax7.hist(processing_times, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Processing Time (hours)')
    ax7.set_ylabel('Number of Jobs')
    ax7.set_title('Processing Time Distribution', fontsize=12, weight='bold')
    ax7.axvline(x=np.mean(processing_times), color='red', linestyle='--', 
                label=f'Avg: {np.mean(processing_times):.1f}h')
    ax7.axvline(x=np.median(processing_times), color='green', linestyle='--', 
                label=f'Median: {np.median(processing_times):.1f}h')
    ax7.legend()
    
    plt.suptitle('LLM Scheduling Performance Dashboard', fontsize=16, weight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def main():
    """Schedule all jobs from snapshot and create visualizations."""
    print("=== LLM Production Scheduling ===\n")
    
    # Initialize scheduler
    scheduler = LLMScheduler()
    
    # Schedule all jobs from normal snapshot
    print("Scheduling all jobs from snapshot_normal.json...")
    print("Using batch processing for 295 jobs...")
    print("This may take 2-3 minutes and cost ~$0.05...")
    
    # First, let's do a smaller test to ensure everything works
    print("\nTesting with first 20 jobs...")
    result = scheduler.schedule(
        snapshot_path="/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json",
        strategy="direct",  # Use direct for speed
        max_jobs=20  # Test with 20 jobs first
    )
    
    print(f"\n✓ Scheduling complete!")
    print(f"  - Jobs scheduled: {result['metrics']['total_jobs']}")
    print(f"  - Response time: {result['llm_metadata']['response_time']:.1f}s")
    print(f"  - Cost: ${result['llm_metadata']['cost_estimate']['total_cost']:.4f}")
    
    # Parse scheduled tasks
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
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
    
    # Validate schedule
    print("\nValidating schedule...")
    validator = ScheduleValidator()
    is_valid, violations = validator.validate_schedule(scheduled_tasks)
    print(f"  - Valid: {is_valid}")
    print(f"  - Violations: {len(violations)}")
    
    # Create output directory
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/llm"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Gantt Chart
    print("  - Generating Gantt chart...")
    fig1, ax1, utilization = create_gantt_chart(
        scheduled_tasks, 
        f"LLM Production Schedule - {len(scheduled_tasks)} Jobs",
        save_path=f"{output_dir}/production_gantt_chart.png"
    )
    plt.close(fig1)
    print("    ✓ Saved: production_gantt_chart.png")
    
    # 2. Metrics Dashboard
    print("  - Generating metrics dashboard...")
    fig2 = create_metrics_dashboard(
        result, 
        scheduled_tasks, 
        violations,
        save_path=f"{output_dir}/production_metrics_dashboard.png"
    )
    plt.close(fig2)
    print("    ✓ Saved: production_metrics_dashboard.png")
    
    # Save detailed results
    print("  - Saving detailed results...")
    summary = {
        'timestamp': datetime.now().isoformat(),
        'snapshot': 'snapshot_normal.json',
        'strategy': 'chain_of_thought',
        'metrics': result['metrics'],
        'llm_metadata': result['llm_metadata'],
        'validation': {
            'is_valid': is_valid,
            'violation_count': len(violations),
            'violation_types': list(set(v.type for v in violations)) if violations else []
        },
        'utilization': {
            'average': np.mean(list(utilization.values())),
            'min': min(utilization.values()),
            'max': max(utilization.values())
        }
    }
    
    with open(f"{output_dir}/production_schedule_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("    ✓ Saved: production_schedule_summary.json")
    
    print(f"\n=== Complete! ===")
    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    main()