import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

def load_production_data():
    """Load the production data snapshot."""
    with open('data/full_production_snapshot.json', 'r') as f:
        return json.load(f)

def create_clean_schedule_visualization():
    """Create a clean Gantt chart visualization of the Phase 4 schedule."""
    
    # Create environment with real data
    print("Creating environment...")
    env = FullProductionEnv(
        snapshot_file='data/full_production_snapshot.json',
        n_machines=152,
        n_jobs=172
    )
    
    # Load the trained Phase 4 model
    print("Loading Phase 4 model...")
    model = PPO.load("models/full_production/final_model.zip")
    
    # Generate schedule
    print("Generating schedule...")
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    # Extract schedule for visualization
    schedule = env.get_schedule()
    
    # Create clean Gantt chart
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Color scheme for different job priorities
    priority_colors = {
        0: '#2E86AB',  # Low priority - Blue
        1: '#A23B72',  # Medium priority - Purple  
        2: '#F18F01',  # High priority - Orange
        3: '#C73E1D'   # Critical priority - Red
    }
    
    # Group jobs by priority for better visualization
    jobs_by_priority = {}
    for job_info in schedule:
        priority = job_info.get('priority', 0)
        if priority not in jobs_by_priority:
            jobs_by_priority[priority] = []
        jobs_by_priority[priority].append(job_info)
    
    # Plot jobs
    y_pos = 0
    job_positions = {}
    y_labels = []
    
    # Sort priorities from highest to lowest
    for priority in sorted(jobs_by_priority.keys(), reverse=True):
        jobs = jobs_by_priority[priority]
        
        # Sort jobs by start time within priority
        jobs.sort(key=lambda x: x['start'])
        
        for job_info in jobs:
            job_id = job_info['job_id']
            start = job_info['start']
            duration = job_info['duration']
            machine = job_info['machine']
            
            # Create rectangle for the job
            rect = patches.Rectangle(
                (start, y_pos), 
                duration, 
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=priority_colors[priority],
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add job ID text
            if duration > 2:  # Only add text if bar is wide enough
                ax.text(
                    start + duration/2, 
                    y_pos + 0.4,
                    f"{job_id}", 
                    ha='center', 
                    va='center',
                    fontsize=8,
                    weight='bold'
                )
            
            y_labels.append(f"{job_id} (M{machine})")
            job_positions[job_id] = y_pos
            y_pos += 1
    
    # Set axis properties
    ax.set_ylim(-0.5, y_pos - 0.5)
    ax.set_xlim(0, max([j['start'] + j['duration'] for j in schedule]) * 1.05)
    
    # Set labels
    ax.set_xlabel('Time (hours)', fontsize=12, weight='bold')
    ax.set_ylabel('Jobs', fontsize=12, weight='bold')
    ax.set_title('Phase 4 Production Schedule - Job View (Clean)', fontsize=16, weight='bold')
    
    # Set y-axis labels
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=priority_colors[3], edgecolor='black', label='Critical Priority'),
        patches.Patch(facecolor=priority_colors[2], edgecolor='black', label='High Priority'),
        patches.Patch(facecolor=priority_colors[1], edgecolor='black', label='Medium Priority'),
        patches.Patch(facecolor=priority_colors[0], edgecolor='black', label='Low Priority')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add statistics
    total_jobs = len(schedule)
    makespan = max([j['start'] + j['duration'] for j in schedule])
    completion_rate = sum(1 for j in schedule if j['start'] + j['duration'] <= makespan) / total_jobs * 100
    
    stats_text = f"Total Jobs: {total_jobs} | Makespan: {makespan:.1f}h | Completion: {completion_rate:.0f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = 'visualizations/phase_4/schedule_job_clean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved clean schedule visualization to {output_path}")
    plt.close()
    
    return schedule, makespan

if __name__ == "__main__":
    schedule, makespan = create_clean_schedule_visualization()
    print(f"\nSchedule generated successfully!")
    print(f"Total makespan: {makespan:.2f} hours")
    print(f"Total jobs scheduled: {len(schedule)}")