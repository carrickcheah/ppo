"""
Test production run with real jobs and visualization.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv
# Remove unused import

# Create visualization directory
viz_dir = Path("/Users/carrickcheah/Project/ppo/visualizations")
viz_dir.mkdir(parents=True, exist_ok=True)

def create_gantt_chart(schedule_data, save_path):
    """Create a Gantt chart visualization of the schedule."""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Colors for different product types
    colors = {
        'CF': '#FF6B6B',
        'CH': '#4ECDC4',
        'CM': '#45B7D1',
        'CP': '#96CEB4',
        'other': '#DDD'
    }
    
    # Plot each job
    for idx, job in enumerate(schedule_data):
        start = job['start_time']
        duration = job['processing_time']
        machine = job['machine_id']
        job_type = job['product_type'][:2] if 'product_type' in job else 'other'
        color = colors.get(job_type, colors['other'])
        
        ax.barh(machine, duration, left=start, height=0.8, 
                color=color, edgecolor='black', linewidth=0.5,
                label=job_type if idx < 4 else "")  # Only label first few
        
        # Add job ID text if bar is wide enough
        if duration > 2:
            ax.text(start + duration/2, machine, job['job_id'][:10], 
                   ha='center', va='center', fontsize=6)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Machine ID')
    ax.set_title('Production Schedule Gantt Chart')
    ax.grid(axis='x', alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Product Type')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gantt chart saved to: {save_path}")

def create_utilization_chart(machine_utils, save_path):
    """Create machine utilization visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Bar chart of utilization by machine
    machines = list(range(len(machine_utils)))
    ax1.bar(machines, machine_utils, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Machine ID')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_title('Machine Utilization by ID')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Histogram of utilization distribution
    ax2.hist(machine_utils, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.set_xlabel('Utilization (%)')
    ax2.set_ylabel('Number of Machines')
    ax2.set_title('Distribution of Machine Utilization')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Utilization chart saved to: {save_path}")

def create_performance_summary(stats, save_path):
    """Create a summary visualization of key metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Makespan over time
    times = stats['time_progress']
    ax1.plot(times, label='Current Time', color='blue', linewidth=2)
    ax1.axhline(y=stats['final_makespan'], color='red', linestyle='--', label=f"Final: {stats['final_makespan']:.1f}h")
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Time (hours)')
    ax1.set_title('Makespan Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Jobs scheduled over time
    ax2.plot(stats['jobs_scheduled'], color='green', linewidth=2)
    ax2.axhline(y=stats['total_jobs'], color='red', linestyle='--', label=f"Total: {stats['total_jobs']}")
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Jobs Scheduled')
    ax2.set_title('Job Completion Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Product type distribution
    product_counts = stats['product_distribution']
    ax3.pie(product_counts.values(), labels=product_counts.keys(), autopct='%1.1f%%', 
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('Jobs by Product Type')
    
    # Key metrics summary
    metrics_text = f"""
    Final Makespan: {stats['final_makespan']:.1f} hours
    Total Jobs: {stats['total_jobs']}
    Completion Rate: {stats['completion_rate']:.1%}
    Avg Machine Utilization: {stats['avg_utilization']:.1%}
    Total Machines Used: {stats['machines_used']}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    ax4.axis('off')
    
    plt.suptitle('Production Run Performance Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance summary saved to: {save_path}")

def run_production_test():
    """Run a complete production test with visualization."""
    print("=== Starting Production Test Run ===")
    
    # Load the trained model
    print("Loading trained model...")
    model_path = "app/models/full_production/final_model.zip"
    model = PPO.load(model_path)
    
    # Create environment
    print("Creating production environment...")
    env = FullProductionEnv(
        n_machines=152,
        n_jobs=500,  # Will load actual jobs from data
        state_compression="hierarchical",
        use_break_constraints=True,
        use_holiday_constraints=True,
        seed=42
    )
    
    # Run the model
    print("Running production schedule...")
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Track statistics
    schedule_data = []
    time_progress = []
    jobs_scheduled_progress = []
    step = 0
    
    while not (terminated or truncated):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Store current state
        time_progress.append(env.current_time)
        # Count scheduled jobs
        n_scheduled = sum(1 for job in env.jobs if hasattr(job, 'scheduled_time') and job.get('scheduled_time', -1) >= 0)
        jobs_scheduled_progress.append(n_scheduled)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record scheduling decision if a job was scheduled
        if info.get('job_scheduled', False):
            job_info = {
                'job_id': info.get('job_id', f'job_{step}'),
                'machine_id': info.get('machine_id', action % env.n_machines),
                'start_time': info.get('start_time', env.current_time),
                'processing_time': info.get('processing_time', 1.0),
                'product_type': info.get('product_type', 'Unknown')
            }
            schedule_data.append(job_info)
        
        step += 1
        
        # Progress update
        if step % 50 == 0:
            n_scheduled = sum(1 for job in env.jobs if hasattr(job, 'scheduled_time') and job.get('scheduled_time', -1) >= 0)
            print(f"Step {step}: Time={env.current_time:.1f}h, Jobs={n_scheduled}/{len(env.jobs)}")
    
    # Final statistics
    print(f"\n=== Schedule Complete ===")
    print(f"Total steps: {step}")
    print(f"Makespan: {info.get('makespan', env.current_time):.1f} hours")
    n_final_scheduled = sum(1 for job in env.jobs if hasattr(job, 'scheduled_time') and job.get('scheduled_time', -1) >= 0)
    print(f"Jobs scheduled: {n_final_scheduled}/{len(env.jobs)}")
    
    # Calculate machine utilization
    machine_utils = []
    for m_id in range(env.n_machines):
        machine_jobs = [j for j in schedule_data if j['machine_id'] == m_id]
        total_time = sum(j['processing_time'] for j in machine_jobs)
        utilization = (total_time / env.current_time * 100) if env.current_time > 0 else 0
        machine_utils.append(utilization)
    
    # Product distribution
    product_types = {}
    for job in env.jobs:
        p_type = job.get('product_type', 'Unknown')[:2]
        product_types[p_type] = product_types.get(p_type, 0) + 1
    
    # Compile statistics
    stats = {
        'final_makespan': info.get('makespan', env.current_time),
        'total_jobs': len(env.jobs),
        'completion_rate': n_final_scheduled / len(env.jobs),
        'avg_utilization': np.mean(machine_utils),
        'machines_used': sum(1 for u in machine_utils if u > 0),
        'time_progress': time_progress,
        'jobs_scheduled': jobs_scheduled_progress,
        'product_distribution': product_types
    }
    
    # Generate visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating visualizations...")
    
    # 1. Gantt Chart
    gantt_path = viz_dir / f"production_gantt_{timestamp}.png"
    create_gantt_chart(schedule_data[:100], gantt_path)  # First 100 jobs for clarity
    
    # 2. Utilization Chart
    util_path = viz_dir / f"machine_utilization_{timestamp}.png"
    create_utilization_chart(machine_utils, util_path)
    
    # 3. Performance Summary
    summary_path = viz_dir / f"performance_summary_{timestamp}.png"
    create_performance_summary(stats, summary_path)
    
    # Save raw data
    data_path = viz_dir / f"production_run_data_{timestamp}.json"
    with open(data_path, 'w') as f:
        json.dump({
            'stats': stats,
            'schedule': schedule_data[:100],  # Sample for file size
            'machine_utilization': machine_utils
        }, f, indent=2)
    print(f"Raw data saved to: {data_path}")
    
    print(f"\n=== Test Complete ===")
    print(f"All visualizations saved to: {viz_dir}")
    
    return stats, schedule_data

if __name__ == "__main__":
    stats, schedule = run_production_test()