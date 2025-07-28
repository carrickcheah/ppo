"""
Test Foundation Models and Generate Schedule Visualizations
Uses the clean_data files and foundation models
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

print("Testing Foundation Models with Clean Data")
print("="*60)

# Configuration
stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
checkpoint_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation"
output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
os.makedirs(output_dir, exist_ok=True)

# Color scheme for visualizations
colors = {
    'on_time': '#4CAF50',      # Green
    'warning': '#FF9800',      # Orange  
    'late': '#F44336',         # Red
    'multi_machine': '#2196F3', # Blue
    'unavailable': '#9E9E9E'   # Gray
}

def test_model_and_visualize(stage_name: str):
    """Test a model and create visualization."""
    print(f"\nProcessing {stage_name}...")
    
    # Check if model exists
    model_path = os.path.join(checkpoint_dir, stage_name, "final_model.zip")
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    # Run one episode to generate schedule
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = 200
    
    # Track scheduling decisions
    schedule_data = {
        'jobs': [],
        'machines': {},
        'timeline': [],
        'metrics': {}
    }
    
    # Initialize machine schedules
    for machine_id in env.machine_ids:
        schedule_data['machines'][machine_id] = []
    
    while not done and steps < max_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        steps += 1
        
        # Record scheduling decision if valid
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            job_info = info.get('scheduled_job_info', {})
            schedule_data['timeline'].append({
                'step': steps,
                'job': info.get('scheduled_job', 'Unknown'),
                'machine': job_info.get('machine', 'Unknown'),
                'start': job_info.get('start_time', 0),
                'duration': job_info.get('processing_time', 0),
                'family': job_info.get('family', 'Unknown')
            })
    
    # Extract final metrics
    schedule_data['metrics'] = {
        'total_jobs': env.total_tasks,
        'scheduled_jobs': len(env.scheduled_jobs),
        'scheduling_rate': len(env.scheduled_jobs) / env.total_tasks if env.total_tasks > 0 else 0,
        'total_steps': steps,
        'families': len(env.families)
    }
    
    print(f"  Scheduled: {schedule_data['metrics']['scheduled_jobs']}/{schedule_data['metrics']['total_jobs']} jobs")
    print(f"  Rate: {schedule_data['metrics']['scheduling_rate']:.1%}")
    print(f"  Steps taken: {steps}")
    
    # Create visualization
    create_gantt_chart(stage_name, schedule_data, env)
    
    return schedule_data

def create_gantt_chart(stage_name: str, schedule_data: dict, env):
    """Create a simple Gantt chart visualization."""
    if not schedule_data['timeline']:
        print(f"  No jobs scheduled for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get machine names for y-axis
    machine_names = [f"Machine {i}" for i in range(len(env.machine_ids))]
    y_positions = {env.machine_ids[i]: i for i in range(len(env.machine_ids))}
    
    # Plot each scheduled job
    for job in schedule_data['timeline']:
        if job['machine'] in y_positions:
            y_pos = y_positions[job['machine']]
            
            # Determine color based on deadline (simplified)
            color = colors['on_time']
            
            # Draw rectangle for job
            rect = plt.Rectangle(
                (job['start'], y_pos - 0.4),
                job['duration'],
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            # Add job label
            if job['duration'] > 2:  # Only label if wide enough
                ax.text(
                    job['start'] + job['duration']/2,
                    y_pos,
                    job['job'][:10],
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white'
                )
    
    # Set labels and title
    ax.set_yticks(range(len(machine_names)))
    ax.set_yticklabels(machine_names)
    ax.set_xlabel('Time (hours)')
    ax.set_title(f'{stage_name.upper()} - Schedule Visualization\n' + 
                 f'Scheduled: {schedule_data["metrics"]["scheduled_jobs"]}/{schedule_data["metrics"]["total_jobs"]} ' +
                 f'({schedule_data["metrics"]["scheduling_rate"]:.1%})')
    
    # Set axis limits
    if schedule_data['timeline']:
        max_time = max(job['start'] + job['duration'] for job in schedule_data['timeline'])
        ax.set_xlim(0, max_time * 1.1)
    else:
        ax.set_xlim(0, 100)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['on_time'], label='Scheduled Job')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, f"{stage_name}_schedule.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to: {output_path}")

# Process all foundation stages
results = {}
for stage in stages:
    result = test_model_and_visualize(stage)
    if result:
        results[stage] = result

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for stage, data in results.items():
    metrics = data['metrics']
    print(f"{stage:12} - Rate: {metrics['scheduling_rate']:6.1%} ({metrics['scheduled_jobs']}/{metrics['total_jobs']} jobs)")

print("\nVisualization files saved to:", output_dir)