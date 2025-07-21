"""Generate clean schedule visualization for Phase 4 model."""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

def create_visualization():
    """Create clean schedule visualization using Phase 4 model."""
    
    print("Loading Phase 4 environment and model...")
    
    # Create environment with parsed production data
    env = FullProductionEnv(
        data_file='data/parsed_production_data.json',
        snapshot_file='data/full_production_snapshot.json',
        n_machines=152,
        n_jobs=172,
        state_compression="hierarchical"
    )
    
    # Load trained model
    model = PPO.load("models/full_production/final_model.zip")
    
    print("Generating schedule...")
    obs, _ = env.reset()
    done = False
    
    # Run episode
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    # Get final schedule
    schedule = env.get_schedule()
    print(f"Scheduled {len(schedule)} jobs")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Color palette for priorities
    colors = {
        0: '#1f77b4',  # Low - Blue
        1: '#ff7f0e',  # Medium - Orange  
        2: '#d62728',  # High - Red
        3: '#9467bd'   # Critical - Purple
    }
    
    # Sort schedule by job priority (highest first) then by start time
    sorted_schedule = sorted(schedule, key=lambda x: (-x.get('priority', 0), x['start']))
    
    # Plot each job
    y_pos = 0
    y_labels = []
    
    for job in sorted_schedule:
        job_id = job['job_id']
        start = job['start']
        duration = job['duration']
        machine = job['machine']
        priority = job.get('priority', 0)
        
        # Draw rectangle
        rect = patches.Rectangle(
            (start, y_pos),
            duration,
            0.9,
            linewidth=1,
            edgecolor='black',
            facecolor=colors.get(priority, '#1f77b4'),
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add text if space allows
        if duration > 1:
            ax.text(
                start + duration/2,
                y_pos + 0.45,
                f"{job_id}",
                ha='center',
                va='center',
                fontsize=7,
                weight='bold',
                color='white' if priority >= 2 else 'black'
            )
        
        y_labels.append(f"{job_id} (M{machine})")
        y_pos += 1
    
    # Configure axes
    ax.set_xlim(0, max(j['start'] + j['duration'] for j in schedule) * 1.02)
    ax.set_ylim(-0.5, len(schedule) - 0.5)
    
    # Labels and title
    ax.set_xlabel('Time (hours)', fontsize=14, weight='bold')
    ax.set_ylabel('Jobs (sorted by priority)', fontsize=14, weight='bold')
    ax.set_title('Phase 4 Full Production Schedule - Clean Job View', fontsize=18, weight='bold', pad=20)
    
    # Y-axis
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[3], edgecolor='black', label='Critical Priority'),
        Patch(facecolor=colors[2], edgecolor='black', label='High Priority'),
        Patch(facecolor=colors[1], edgecolor='black', label='Medium Priority'),
        Patch(facecolor=colors[0], edgecolor='black', label='Low Priority')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add performance metrics
    makespan = max(j['start'] + j['duration'] for j in schedule)
    completion_rate = len(schedule) / 172 * 100
    
    metrics_text = (
        f"Total Jobs: {len(schedule)}/172\n"
        f"Makespan: {makespan:.1f} hours\n"
        f"Completion: {completion_rate:.1f}%"
    )
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    output_path = 'visualizations/phase_4/schedule_job_clean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    print(f"Makespan: {makespan:.1f} hours")
    print(f"Jobs scheduled: {len(schedule)}/172")
    
    plt.close()

if __name__ == "__main__":
    create_visualization()