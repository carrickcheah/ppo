"""
Create clean, simple Gantt charts for curriculum learning results.
Similar style to schedule_job_clean.png - minimal and professional.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv
from stable_baselines3 import PPO


def collect_schedule_data(env, model=None, n_steps=200):
    """Run the environment and collect scheduling data."""
    obs, _ = env.reset()
    
    schedule_data = []
    done = False
    step = 0
    
    while not done and step < n_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Random valid action
            if env.valid_actions:
                action = np.random.randint(len(env.valid_actions))
            else:
                break
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect scheduling info
        if 'scheduled_job' in info and 'start_time' in info:
            # Extract machine ID
            machine_info = info['on_machine']
            if isinstance(machine_info, str):
                machine_id = int(machine_info.split()[0])
                machine_name = machine_info.split('(')[1].rstrip(')')
            else:
                machine_id = int(machine_info)
                machine_name = f"M{machine_id}"
                
            # Parse job info to get full name
            job_id = info['scheduled_job']
            family_id = job_id.split('-')[0]
            
            # Try to get product code from environment
            product_code = "Unknown"
            if hasattr(env, 'families_data') and family_id in env.families_data:
                product_code = env.families_data[family_id].get('product', 'Unknown')
            
            full_job_name = f"JOST{family_id}_{product_code}-{job_id.split('-')[1]}"
                
            schedule_data.append({
                'job_id': job_id,
                'full_job_name': full_job_name,
                'machine_id': machine_id,
                'machine_name': machine_name,
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'processing_time': info['processing_time'],
                'setup_time': info.get('setup_time', 0),
                'is_important': info.get('is_important', False)
            })
        
        step += 1
    
    return schedule_data, env.episode_makespan


def create_clean_job_gantt(schedule_data, makespan, save_path=None):
    """Create a clean Gantt chart with jobs on Y-axis."""
    if not schedule_data:
        print("No schedule data to visualize")
        return
    
    # Sort by start time
    schedule_data_sorted = sorted(schedule_data, key=lambda x: x['start_time'])
    
    # Select first 50 jobs for clarity
    schedule_data_display = schedule_data_sorted[:50]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Simple color scheme - red for important, blue for normal
    colors = {
        True: '#FF6B6B',   # Red for important
        False: '#4ECDC4'   # Blue for normal
    }
    
    # Plot jobs
    for i, job in enumerate(schedule_data_display):
        # Main job bar
        rect = patches.Rectangle(
            (job['start_time'], i),
            job['processing_time'] + job['setup_time'],
            0.8,
            facecolor=colors[job['is_important']],
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Set labels
    job_labels = [job['full_job_name'] for job in schedule_data_display]
    ax.set_yticks(range(len(schedule_data_display)))
    ax.set_yticklabels(job_labels, fontsize=8)
    
    # Timeline on x-axis
    ax.set_xlim(0, makespan)
    ax.set_ylim(-1, len(schedule_data_display))
    
    # Format x-axis to show hours
    ax.set_xlabel('Timeline (hours)', fontsize=10)
    ax.set_title('Production Schedule', fontsize=12, fontweight='bold')
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_clean_machine_gantt(schedule_data, makespan, n_machines=40, save_path=None):
    """Create a clean Gantt chart with machines on Y-axis."""
    if not schedule_data:
        print("No schedule data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Simple color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48C9B0', '#6C5CE7', '#A29BFE']
    
    # Group by families for coloring
    family_colors = {}
    color_idx = 0
    
    for job in schedule_data:
        family_id = job['job_id'].split('-')[0]
        if family_id not in family_colors:
            family_colors[family_id] = colors[color_idx % len(colors)]
            color_idx += 1
        
        # Plot job
        rect = patches.Rectangle(
            (job['start_time'], job['machine_id']),
            job['processing_time'] + job['setup_time'],
            0.8,
            facecolor=family_colors[family_id],
            edgecolor='white',
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        # Add job text if it fits
        if job['processing_time'] > 0.5:
            ax.text(
                job['start_time'] + (job['processing_time'] + job['setup_time']) / 2,
                job['machine_id'] + 0.4,
                job['job_id'],
                ha='center',
                va='center',
                fontsize=6,
                color='white',
                fontweight='bold'
            )
    
    # Machine labels
    ax.set_ylim(-0.5, n_machines - 0.5)
    ax.set_yticks(range(0, n_machines, 2))
    ax.set_yticklabels([f'Machine {i}' for i in range(0, n_machines, 2)], fontsize=9)
    
    # Timeline
    ax.set_xlim(0, makespan)
    ax.set_xlabel('Timeline (hours)', fontsize=10)
    ax.set_title('Machine Schedule', fontsize=12, fontweight='bold')
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add utilization text
    machine_busy = {}
    for job in schedule_data:
        if job['machine_id'] not in machine_busy:
            machine_busy[job['machine_id']] = 0
        machine_busy[job['machine_id']] += job['processing_time'] + job['setup_time']
    
    avg_util = np.mean([busy/makespan for busy in machine_busy.values()]) * 100
    ax.text(0.02, 0.98, f'Average Utilization: {avg_util:.1f}%', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Generate clean curriculum visualizations."""
    print("Generating Clean Curriculum Visualizations...")
    print("=" * 60)
    
    # Create environment without breaks
    env = ScaledProductionEnv(
        n_machines=40,
        use_break_constraints=False,
        seed=42
    )
    
    # Collect schedule data (using random policy for now)
    print("Collecting schedule data...")
    schedule_data, makespan = collect_schedule_data(env, model=None)
    
    if not schedule_data:
        print("No jobs were scheduled!")
        return
    
    print(f"Scheduled {len(schedule_data)} jobs")
    print(f"Makespan: {makespan:.1f} hours")
    
    # Create visualizations
    save_dir = Path('visualizations/curriculum_clean/')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Clean job Gantt
    print("\nCreating clean job Gantt chart...")
    create_clean_job_gantt(
        schedule_data,
        makespan,
        save_path=save_dir / 'schedule_job_clean.png'
    )
    print(f"Saved to: {save_dir / 'schedule_job_clean.png'}")
    
    # 2. Clean machine Gantt
    print("\nCreating clean machine Gantt chart...")
    create_clean_machine_gantt(
        schedule_data,
        makespan,
        n_machines=40,
        save_path=save_dir / 'schedule_machine_clean.png'
    )
    print(f"Saved to: {save_dir / 'schedule_machine_clean.png'}")
    
    print("\n" + "=" * 60)
    print("Clean visualizations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()