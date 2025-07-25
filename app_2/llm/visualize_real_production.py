"""
Visualize REAL production schedule with focus on multi-machine jobs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime
import json
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_scheduler import LLMScheduler
from llm.parser import ScheduleParser, ScheduledTask
from llm.validator import ScheduleValidator


def create_multi_machine_gantt(scheduled_tasks, title="Real Production Schedule - Multi-Machine Jobs"):
    """Create Gantt chart highlighting multi-machine jobs."""
    
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Separate single and multi-machine jobs
    single_machine_jobs = [t for t in scheduled_tasks if len(t.machine_ids) == 1]
    multi_machine_jobs = [t for t in scheduled_tasks if len(t.machine_ids) > 1]
    
    # Color scheme
    single_color = 'lightblue'
    multi_colors = sns.color_palette("husl", len(multi_machine_jobs))
    
    # Group by machine
    tasks_by_machine = {}
    for task in scheduled_tasks:
        for machine_id in task.machine_ids:
            if machine_id not in tasks_by_machine:
                tasks_by_machine[machine_id] = []
            tasks_by_machine[machine_id].append(task)
    
    machine_ids = sorted(tasks_by_machine.keys())
    
    # Time bounds
    all_starts = [task.start_time for task in scheduled_tasks]
    all_ends = [task.end_time for task in scheduled_tasks]
    min_time = min(all_starts)
    max_time = max(all_ends)
    
    # Plot
    y_pos = 0
    y_labels = []
    multi_job_colors = {job.job_id: multi_colors[i] for i, job in enumerate(multi_machine_jobs)}
    
    for machine_id in machine_ids:
        y_labels.append(f"M{machine_id}")
        
        for task in sorted(tasks_by_machine[machine_id], key=lambda t: t.start_time):
            start_offset = (task.start_time - min_time).total_seconds() / 3600
            duration = (task.end_time - task.start_time).total_seconds() / 3600
            
            # Color based on job type
            if len(task.machine_ids) > 1:
                color = multi_job_colors[task.job_id]
                edge_width = 2
                alpha = 0.9
            else:
                color = single_color
                edge_width = 0.5
                alpha = 0.6
            
            rect = Rectangle(
                (start_offset, y_pos - 0.4),
                duration,
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=edge_width,
                alpha=alpha
            )
            ax.add_patch(rect)
            
            # Label multi-machine jobs
            if len(task.machine_ids) > 1 and duration > 2:
                label = f"{len(task.machine_ids)} machines"
                ax.text(
                    start_offset + duration/2,
                    y_pos,
                    label,
                    ha='center',
                    va='center',
                    fontsize=7,
                    weight='bold'
                )
        
        y_pos += 1
    
    # Highlight multi-machine job info
    info_text = f"Multi-Machine Jobs: {len(multi_machine_jobs)}\n"
    for i, job in enumerate(multi_machine_jobs[:3]):  # Show first 3
        info_text += f"• {job.job_id}: {len(job.machine_ids)} machines\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Formatting
    ax.set_ylim(-0.5, len(machine_ids) - 0.5)
    ax.set_xlim(0, (max_time - min_time).total_seconds() / 3600)
    ax.set_yticks(range(len(machine_ids)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Time markers
    time_marks = range(0, int((max_time - min_time).total_seconds() / 3600) + 1, 24)
    ax.set_xticks(time_marks)
    ax.set_xticklabels([f"Day {t//24}" for t in time_marks])
    
    plt.tight_layout()
    return fig, ax


def create_constraint_validation_chart(violations, scheduled_tasks):
    """Create chart showing constraint validation results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Constraint satisfaction pie chart
    # Count satisfied constraints (each task can have multiple constraints)
    total_constraints = len(scheduled_tasks) * 4  # Approx 4 constraints per task
    violated_constraints = len(violations)
    satisfied_constraints = max(0, total_constraints - violated_constraints)
    
    sizes = [satisfied_constraints, violated_constraints]
    labels = ['Satisfied', 'Violated']
    colors = ['green', 'red']
    
    if all(s >= 0 for s in sizes) and sum(sizes) > 0:
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    else:
        ax1.text(0.5, 0.5, f'{violated_constraints} Violations', ha='center', va='center', 
                fontsize=14, color='red', weight='bold', transform=ax1.transAxes)
    ax1.set_title('Overall Constraint Satisfaction', fontsize=12, weight='bold')
    
    # 2. Violation types
    if violations:
        violation_types = {}
        for v in violations:
            vtype = v.type.split('_')[0]
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        ax2.bar(violation_types.keys(), violation_types.values(), color='coral')
        ax2.set_title('Violations by Type', fontsize=12, weight='bold')
        ax2.set_ylabel('Count')
    else:
        ax2.text(0.5, 0.5, 'No Violations!', ha='center', va='center', 
                fontsize=20, color='green', weight='bold', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # 3. Multi-machine job statistics
    multi_jobs = [t for t in scheduled_tasks if len(t.machine_ids) > 1]
    machine_counts = [len(t.machine_ids) for t in multi_jobs]
    
    if machine_counts:
        ax3.hist(machine_counts, bins=range(1, max(machine_counts)+2), color='purple', alpha=0.7)
        ax3.set_xlabel('Number of Machines Required')
        ax3.set_ylabel('Number of Jobs')
        ax3.set_title(f'Multi-Machine Jobs ({len(multi_jobs)} total)', fontsize=12, weight='bold')
        ax3.set_xticks(range(2, max(machine_counts)+1))
    else:
        ax3.text(0.5, 0.5, 'No Multi-Machine Jobs', ha='center', va='center', 
                fontsize=14, transform=ax3.transAxes)
    
    # 4. Sequence compliance
    families = {}
    for task in scheduled_tasks:
        if task.family_id not in families:
            families[task.family_id] = []
        families[task.family_id].append(task)
    
    sequence_correct = 0
    sequence_total = 0
    
    for family_id, tasks in families.items():
        tasks_sorted = sorted(tasks, key=lambda t: t.sequence)
        sequence_total += len(tasks)
        
        # Check if sequences are in correct order by time
        correct = True
        for i in range(1, len(tasks_sorted)):
            if tasks_sorted[i].start_time < tasks_sorted[i-1].end_time:
                correct = False
                break
        
        if correct:
            sequence_correct += len(tasks)
    
    sequence_data = [sequence_correct, sequence_total - sequence_correct]
    labels = ['Correct Sequence', 'Sequence Issues']
    colors = ['lightgreen', 'lightcoral']
    
    ax4.pie(sequence_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Sequence Constraint Compliance', fontsize=12, weight='bold')
    
    plt.suptitle('Real Production Constraint Validation', fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Schedule and visualize real production data."""
    print("=== REAL PRODUCTION SCHEDULING WITH LLM ===\n")
    
    scheduler = LLMScheduler()
    
    # Schedule with focus on multi-machine jobs
    print("Scheduling real factory jobs...")
    print("Including jobs that require up to 21 machines simultaneously...\n")
    
    result = scheduler.schedule(
        snapshot_path="/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json",
        strategy="direct",  # Faster strategy
        max_jobs=15  # Smaller batch for quicker results
    )
    
    print(f"✓ Scheduled {result['metrics']['total_jobs']} real production jobs")
    print(f"  - Response time: {result['llm_metadata']['response_time']:.1f}s")
    print(f"  - Cost: ${result['llm_metadata']['cost_estimate']['total_cost']:.4f}")
    
    # Parse tasks
    scheduled_tasks = []
    for family_id, tasks in result['schedule'].items():
        for task_data in tasks:
            from datetime import datetime
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
    
    # Analyze multi-machine jobs
    multi_jobs = [t for t in scheduled_tasks if len(t.machine_ids) > 1]
    print(f"\nMulti-machine jobs: {len(multi_jobs)}")
    for job in multi_jobs[:5]:
        print(f"  - {job.job_id}: {len(job.machine_ids)} machines")
    
    # Validate
    validator = ScheduleValidator()
    is_valid, violations = validator.validate_schedule(scheduled_tasks)
    print(f"\nConstraint validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Violations: {len(violations)}")
    
    # Create visualizations
    output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/llm"
    os.makedirs(output_dir, exist_ok=True)
    
    # Multi-machine Gantt chart
    fig1, _ = create_multi_machine_gantt(scheduled_tasks)
    fig1.savefig(f"{output_dir}/real_production_gantt.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n✓ Saved: real_production_gantt.png")
    
    # Constraint validation chart
    fig2 = create_constraint_validation_chart(violations, scheduled_tasks)
    fig2.savefig(f"{output_dir}/constraint_validation.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved: constraint_validation.png")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_jobs': len(scheduled_tasks),
        'multi_machine_jobs': len(multi_jobs),
        'largest_multi_machine': max([len(t.machine_ids) for t in multi_jobs]) if multi_jobs else 0,
        'constraint_violations': len(violations),
        'is_valid': is_valid,
        'real_data': True,
        'snapshot': 'snapshot_normal.json'
    }
    
    with open(f"{output_dir}/real_production_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== COMPLETE ===")
    print(f"All visualizations saved to: {output_dir}")
    print("\nThis schedule uses:")
    print("✓ REAL job IDs from customer orders")
    print("✓ REAL machine IDs from factory floor")
    print("✓ REAL processing times from production data")
    print("✓ REAL multi-machine constraints")


if __name__ == "__main__":
    main()