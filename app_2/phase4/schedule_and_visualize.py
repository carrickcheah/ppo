#!/usr/bin/env python3
"""
Schedule jobs using Phase 4 PPO models and create professional visualizations.
Creates both job allocation (job-view) and machine allocation (machine-view) Gantt charts.
Uses ONLY trained PPO models for scheduling - no fallback heuristics.
"""

import json
import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from stable_baselines3 import PPO

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from environments.small_balanced_env import SmallBalancedEnvironment


def load_latest_balanced_model():
    """Load the latest trained balanced strategy model."""
    model_dir = Path(__file__).parent / "results" / "balanced" / "checkpoints"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find all model files and sort by iteration and steps
    model_files = sorted(model_dir.glob("*.zip"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Use the latest model (highest iteration and steps)
    latest_model = model_files[-1]
    print(f"Loading PPO model: {latest_model.name}")
    
    model = PPO.load(str(latest_model))
    return model, latest_model.stem


def schedule_with_ppo_model(model, env):
    """
    Schedule jobs using ONLY the trained PPO model.
    No fallback heuristics - strictly follows PPO model predictions.
    """
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = 500  # Reasonable limit for 75 tasks
    
    scheduled_jobs = []
    machine_schedules = {m['machine_id']: [] for m in env.machines}
    
    print("\nStarting PPO model scheduling...")
    print(f"Total tasks to schedule: {env.total_tasks}")
    print(f"Available machines: {len(env.machines)}")
    
    while not done and steps < max_steps:
        # Get action from PPO model (deterministic for reproducibility)
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Track successful scheduling
        if info.get('action_valid', False):
            # Extract the scheduled job info from the info dict
            if 'scheduled_job' in info and info['scheduled_job']:
                job_data = info['scheduled_job']
                # The job_assignments structure uses different keys
                if hasattr(env, 'job_assignments'):
                    # Find the most recently added job
                    for job_id, assignment in env.job_assignments.items():
                        # Check if this job is already in our list
                        if not any(j['job_idx'] == job_id for j in scheduled_jobs):
                            # Extract machine (first one if multiple)
                            machine_id = assignment['machines'][0] if assignment['machines'] else None
                            
                            if machine_id is not None:
                                job_info = {
                                    'job_idx': job_id,
                                    'machine_id': machine_id,
                                    'start_time': assignment['start'],
                                    'duration': assignment['end'] - assignment['start'],
                                    'end_time': assignment['end']
                                }
                                
                                scheduled_jobs.append(job_info)
                                machine_schedules[machine_id].append(job_info)
                                print(f"Step {steps}: Scheduled {job_id} on machine {machine_id}")
    
    print(f"\nPPO scheduling completed:")
    print(f"- Steps taken: {steps}")
    print(f"- Jobs scheduled: {len(scheduled_jobs)}/{env.total_tasks}")
    print(f"- Completion rate: {len(scheduled_jobs)/env.total_tasks*100:.1f}%")
    
    return scheduled_jobs, machine_schedules, env


def create_job_allocation_chart(scheduled_jobs, env, model_name, save_path):
    """
    Create job allocation Gantt chart (job-view).
    Shows timeline for each job with color coding for status.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare job data
    if not scheduled_jobs:
        # Handle case with no scheduled jobs
        ax.text(0.5, 0.5, 'No jobs scheduled by PPO model', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title(f'Job Allocation Chart - {model_name}')
    else:
        # Get unique job families from data
        with open(Path(__file__).parent / "data" / "small_balanced_data.json", 'r') as f:
            data = json.load(f)
        
        families = list(data['families'].keys())
        family_tasks = {}
        task_to_family = {}
        y_positions = {}
        y_pos = 0
        
        # Map tasks to families and create y-positions
        for family_id in families:
            family_data = data['families'][family_id]
            family_tasks[family_id] = []
            
            for task in family_data['tasks']:
                task_id = task['task_id']
                task_to_family[y_pos] = (family_id, task_id)
                y_positions[y_pos] = y_pos
                family_tasks[family_id].append(task_id)
                y_pos += 1
        
        # Color scheme
        colors = {
            'on_time': '#2ecc71',  # Green
            'late': '#e74c3c',      # Red
            'important': '#9b59b6', # Purple
            'normal': '#3498db'     # Blue
        }
        
        # Plot scheduled jobs
        for job in scheduled_jobs:
            job_idx = job['job_idx']
            if job_idx in y_positions:
                y = y_positions[job_idx]
                
                # Determine color based on job properties
                color = colors['normal']
                
                # Create rectangle for job
                rect = Rectangle(
                    (job['start_time'], y - 0.4),
                    job['duration'],
                    0.8,
                    linewidth=1,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.7
                )
                ax.add_patch(rect)
                
                # Add machine label if space permits
                if job['duration'] > 2:
                    ax.text(
                        job['start_time'] + job['duration']/2,
                        y,
                        f"M{job['machine_id']}",
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white',
                        weight='bold'
                    )
        
        # Set y-axis labels
        y_labels = []
        for y in range(len(y_positions)):
            if y in task_to_family:
                family_id, task_id = task_to_family[y]
                # Shorten long IDs for display
                short_family = family_id[-8:] if len(family_id) > 8 else family_id
                y_labels.append(f"{short_family}_{task_id.split('_')[1]}")
            else:
                y_labels.append(f"Task {y}")
        
        ax.set_yticks(range(len(y_positions)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_ylim(-0.5, len(y_positions) - 0.5)
        
        # Set x-axis (time)
        if scheduled_jobs:
            max_time = max(j['end_time'] for j in scheduled_jobs)
            ax.set_xlim(0, max_time * 1.1)
        else:
            ax.set_xlim(0, 100)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Jobs (Family_Sequence)', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add title
        ax.set_title(f'Job Allocation Chart - PPO Model: {model_name}', fontsize=14, weight='bold')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=colors['normal'], label='Normal Job', alpha=0.7),
            patches.Patch(color=colors['on_time'], label='On-time', alpha=0.7),
            patches.Patch(color=colors['late'], label='Late', alpha=0.7),
            patches.Patch(color=colors['important'], label='Important', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add statistics box
        stats_text = f"PPO Model Performance:\n"
        stats_text += f"Jobs Scheduled: {len(scheduled_jobs)}/{env.total_tasks}\n"
        stats_text += f"Completion Rate: {len(scheduled_jobs)/env.total_tasks*100:.1f}%\n"
        
        if scheduled_jobs:
            makespan = max(j['end_time'] for j in scheduled_jobs)
            stats_text += f"Makespan: {makespan:.1f} hours"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Job allocation chart saved to: {save_path}")
    
    return fig


def create_machine_allocation_chart(scheduled_jobs, machine_schedules, env, model_name, save_path):
    """
    Create machine allocation Gantt chart (machine-view).
    Shows machine utilization over time with job assignments.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Machine names and IDs
    machines = env.machines
    # Machines don't have names in this environment, use IDs
    machine_names = [f"Machine {m['machine_id']}" for m in machines]
    machine_ids = [m['machine_id'] for m in machines]
    
    # Color palette for different jobs
    if scheduled_jobs:
        num_jobs = len(scheduled_jobs)
        colors = plt.cm.Set3(np.linspace(0, 1, max(num_jobs, 10)))
        job_colors = {job['job_idx']: colors[i % len(colors)] 
                     for i, job in enumerate(scheduled_jobs)}
    else:
        job_colors = {}
    
    # Plot jobs on each machine
    for i, machine in enumerate(machines):
        machine_id = machine['machine_id']
        machine_jobs = machine_schedules.get(machine_id, [])
        
        for job in machine_jobs:
            # Create rectangle for job
            rect = Rectangle(
                (job['start_time'], i - 0.4),
                job['duration'],
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=job_colors.get(job['job_idx'], 'gray'),
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add job label if space permits
            if job['duration'] > 3:
                ax.text(
                    job['start_time'] + job['duration']/2,
                    i,
                    f"Job {job['job_idx']}",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    weight='bold'
                )
    
    # Set y-axis (machines)
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machine_names, fontsize=10)
    ax.set_ylim(-0.5, len(machines) - 0.5)
    
    # Set x-axis (time)
    if scheduled_jobs:
        max_time = max(j['end_time'] for j in scheduled_jobs)
        ax.set_xlim(0, max_time * 1.1)
        
        # Add makespan indicator
        ax.axvline(x=max_time, color='green', linestyle='--', 
                  alpha=0.7, label=f'Makespan: {max_time:.1f}h')
    else:
        ax.set_xlim(0, 100)
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add title
    ax.set_title(f'Machine Allocation Chart - PPO Model: {model_name}', 
                fontsize=14, weight='bold')
    
    # Calculate and display utilization
    utilization_text = "Machine Utilization:\n"
    
    if scheduled_jobs:
        makespan = max(j['end_time'] for j in scheduled_jobs)
        
        for i, machine in enumerate(machines):
            machine_id = machine['machine_id']
            machine_jobs = machine_schedules.get(machine_id, [])
            
            if machine_jobs and makespan > 0:
                busy_time = sum(j['duration'] for j in machine_jobs)
                utilization = (busy_time / makespan) * 100
                utilization_text += f"Machine {machine_id}: {utilization:.1f}%\n"
            else:
                utilization_text += f"Machine {machine_id}: 0.0%\n"
        
        # Overall utilization
        total_busy = sum(j['duration'] for j in scheduled_jobs)
        total_capacity = makespan * len(machines)
        
        if total_capacity > 0:
            overall_util = (total_busy / total_capacity) * 100
            utilization_text += f"\nOverall: {overall_util:.1f}%"
    else:
        for machine in machines:
            utilization_text += f"Machine {machine['machine_id']}: 0.0%\n"
        utilization_text += "\nOverall: 0.0%"
    
    # Add utilization box
    ax.text(0.85, 0.98, utilization_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add statistics box
    stats_text = f"PPO Scheduling Results:\n"
    stats_text += f"Jobs Scheduled: {len(scheduled_jobs)}/{env.total_tasks}\n"
    stats_text += f"Machines Used: {len([m for m in machine_schedules.values() if m])}/{len(machines)}\n"
    stats_text += f"Completion Rate: {len(scheduled_jobs)/env.total_tasks*100:.1f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend if there's a makespan line
    if scheduled_jobs:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Machine allocation chart saved to: {save_path}")
    
    return fig


def main():
    """Main function to schedule jobs and create visualizations."""
    print("=" * 70)
    print("Phase 4 PPO Model Scheduling and Visualization")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("/home/azureuser/ppo/app_2/visualizations/phase_4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the latest balanced model
    try:
        model, model_name = load_latest_balanced_model()
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment
    print("\nInitializing Small Balanced Environment...")
    env = SmallBalancedEnvironment(verbose=False)
    print(f"Environment ready: {env.total_tasks} tasks, {len(env.machines)} machines")
    
    # Schedule jobs using PPO model
    scheduled_jobs, machine_schedules, env = schedule_with_ppo_model(model, env)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create job allocation chart
    print("\nGenerating job allocation Gantt chart...")
    job_chart_path = output_dir / f"phase4_job_allocation_gantt_{timestamp}.png"
    job_fig = create_job_allocation_chart(scheduled_jobs, env, model_name, job_chart_path)
    
    # Create machine allocation chart
    print("\nGenerating machine allocation Gantt chart...")
    machine_chart_path = output_dir / f"phase4_machine_allocation_gantt_{timestamp}.png"
    machine_fig = create_machine_allocation_chart(
        scheduled_jobs, machine_schedules, env, model_name, machine_chart_path
    )
    
    # Save scheduling results to JSON
    results_path = output_dir / f"schedule_results_{timestamp}.json"
    
    results = {
        'timestamp': timestamp,
        'model_used': model_name,
        'environment': 'SmallBalancedEnvironment',
        'total_tasks': env.total_tasks,
        'total_machines': len(env.machines),
        'jobs_scheduled': len(scheduled_jobs),
        'completion_rate': len(scheduled_jobs) / env.total_tasks * 100 if env.total_tasks > 0 else 0,
        'scheduled_jobs': scheduled_jobs,
        'machine_schedules': {str(k): v for k, v in machine_schedules.items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nScheduling results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SCHEDULING SUMMARY")
    print("=" * 70)
    print(f"Model Used: {model_name}")
    print(f"Total Tasks: {env.total_tasks}")
    print(f"Jobs Scheduled: {len(scheduled_jobs)}")
    print(f"Completion Rate: {results['completion_rate']:.1f}%")
    print(f"\nVisualization files saved:")
    print(f"- Job Allocation: {job_chart_path}")
    print(f"- Machine Allocation: {machine_chart_path}")
    print(f"- Results JSON: {results_path}")
    print("=" * 70)
    
    # Display the plots
    plt.show()


if __name__ == "__main__":
    main()