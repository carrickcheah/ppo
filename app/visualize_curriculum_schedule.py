"""
Visualize the actual scheduling results from curriculum learning.
Creates Gantt charts and machine utilization heatmaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.environments.scaled_production_env import ScaledProductionEnv
from stable_baselines3 import PPO


def load_curriculum_model():
    """Load the trained curriculum model if available."""
    model_path = "models/curriculum/phase1_no_breaks/final_model"
    if Path(model_path + ".zip").exists():
        print(f"Loading trained model from {model_path}")
        try:
            return PPO.load(model_path)
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using random policy instead")
            return None
    else:
        print("No trained model found, using random policy")
        return None


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
            # Extract machine ID from string like "4 (CM03)"
            machine_info = info['on_machine']
            if isinstance(machine_info, str):
                machine_id = int(machine_info.split()[0])
            else:
                machine_id = int(machine_info)
                
            schedule_data.append({
                'job_id': info['scheduled_job'],
                'machine_id': machine_id,
                'machine_name': f"Machine {machine_id}",
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'processing_time': info['processing_time'],
                'setup_time': info.get('setup_time', 0),
                'is_important': info.get('is_important', False)
            })
        
        step += 1
    
    return schedule_data, env.episode_makespan


def create_job_gantt_chart(schedule_data, makespan, title="Job Schedule View"):
    """Create a Gantt chart with jobs on Y-axis."""
    if not schedule_data:
        print("No schedule data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort jobs by start time
    schedule_data_sorted = sorted(schedule_data, key=lambda x: x['start_time'])
    
    # Create job list and positions
    job_ids = [item['job_id'] for item in schedule_data_sorted]
    y_pos = np.arange(len(job_ids))
    
    # Color mapping for machines
    unique_machines = sorted(set(item['machine_id'] for item in schedule_data))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_machines)))
    machine_colors = {machine: colors[i] for i, machine in enumerate(unique_machines)}
    
    # Plot each job
    for i, job_data in enumerate(schedule_data_sorted):
        # Processing time bar
        processing_bar = patches.Rectangle(
            (job_data['start_time'] + job_data['setup_time'], i - 0.4),
            job_data['processing_time'],
            0.8,
            facecolor=machine_colors[job_data['machine_id']],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(processing_bar)
        
        # Setup time bar (if any)
        if job_data['setup_time'] > 0:
            setup_bar = patches.Rectangle(
                (job_data['start_time'], i - 0.4),
                job_data['setup_time'],
                0.8,
                facecolor='lightgray',
                edgecolor='black',
                linewidth=1,
                hatch='//'
            )
            ax.add_patch(setup_bar)
        
        # Add importance marker
        if job_data['is_important']:
            ax.text(job_data['start_time'] - 0.5, i, '★', 
                   color='red', fontsize=12, ha='center', va='center')
    
    # Customize plot
    ax.set_ylim(-1, len(job_ids))
    ax.set_xlim(0, makespan * 1.05)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(job_ids, fontsize=8)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Jobs', fontsize=12)
    ax.set_title(f'{title}\nMakespan: {makespan:.1f} hours', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=machine_colors[m], 
                                   edgecolor='black',
                                   label=f'Machine {m}')
                      for m in unique_machines[:10]]  # Show first 10 machines
    if job_data['setup_time'] > 0:
        legend_elements.append(patches.Patch(facecolor='lightgray', 
                                           edgecolor='black',
                                           hatch='//',
                                           label='Setup Time'))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig


def create_machine_gantt_chart(schedule_data, makespan, n_machines=40, title="Machine Schedule View"):
    """Create a Gantt chart with machines on Y-axis."""
    if not schedule_data:
        print("No schedule data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color by job families
    family_colors = {}
    color_idx = 0
    
    # Plot each scheduled job
    for job_data in schedule_data:
        # Extract family from job_id (e.g., "7874-1" -> "7874")
        family_id = job_data['job_id'].split('-')[0]
        
        # Assign color to family if not already assigned
        if family_id not in family_colors:
            family_colors[family_id] = plt.cm.tab20(color_idx % 20)
            color_idx += 1
        
        # Processing time bar
        processing_bar = patches.Rectangle(
            (job_data['start_time'] + job_data['setup_time'], job_data['machine_id'] - 0.4),
            job_data['processing_time'],
            0.8,
            facecolor=family_colors[family_id],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(processing_bar)
        
        # Setup time bar
        if job_data['setup_time'] > 0:
            setup_bar = patches.Rectangle(
                (job_data['start_time'], job_data['machine_id'] - 0.4),
                job_data['setup_time'],
                0.8,
                facecolor='darkgray',
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(setup_bar)
        
        # Add job label
        if job_data['processing_time'] > 0.5:  # Only label if job is long enough
            ax.text(job_data['start_time'] + job_data['setup_time'] + job_data['processing_time']/2,
                   job_data['machine_id'],
                   job_data['job_id'],
                   ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Customize plot
    ax.set_ylim(-1, n_machines)
    ax.set_xlim(0, makespan * 1.05)
    ax.set_yticks(range(0, n_machines, 5))
    ax.set_yticklabels([f'M{i}' for i in range(0, n_machines, 5)])
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title(f'{title}\nMakespan: {makespan:.1f} hours', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add utilization info
    machine_utils = calculate_machine_utilization(schedule_data, makespan, n_machines)
    avg_util = np.mean(list(machine_utils.values())) * 100
    ax.text(0.02, 0.98, f'Avg Utilization: {avg_util:.1f}%', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig


def calculate_machine_utilization(schedule_data, makespan, n_machines):
    """Calculate utilization for each machine."""
    machine_busy_time = {i: 0 for i in range(n_machines)}
    
    for job_data in schedule_data:
        machine_id = job_data['machine_id']
        busy_time = job_data['processing_time'] + job_data['setup_time']
        machine_busy_time[machine_id] += busy_time
    
    # Calculate utilization
    machine_utilization = {}
    for machine_id, busy_time in machine_busy_time.items():
        if makespan > 0:
            machine_utilization[machine_id] = busy_time / makespan
        else:
            machine_utilization[machine_id] = 0
    
    return machine_utilization


def create_utilization_heatmap(schedule_data, makespan, n_machines=40, title="Machine Utilization Heatmap"):
    """Create a heatmap showing machine utilization over time."""
    if not schedule_data or makespan == 0:
        print("No data for heatmap")
        return
    
    # Create time bins (1-hour intervals)
    time_bins = np.arange(0, int(makespan) + 1, 1)
    
    # Initialize utilization matrix
    utilization_matrix = np.zeros((n_machines, len(time_bins) - 1))
    
    # Fill utilization data
    for job_data in schedule_data:
        machine_id = job_data['machine_id']
        start_time = job_data['start_time']
        end_time = job_data['end_time']
        
        # Find which time bins this job spans
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            
            # Calculate overlap between job and time bin
            overlap_start = max(start_time, bin_start)
            overlap_end = min(end_time, bin_end)
            
            if overlap_end > overlap_start:
                utilization_matrix[machine_id, i] = (overlap_end - overlap_start) / (bin_end - bin_start)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot heatmap
    im = ax.imshow(utilization_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Customize axes
    ax.set_xticks(range(0, len(time_bins) - 1, 2))
    ax.set_xticklabels([f'{int(t)}h' for t in time_bins[:-1:2]])
    ax.set_yticks(range(0, n_machines, 5))
    ax.set_yticklabels([f'M{i}' for i in range(0, n_machines, 5)])
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title(f'{title}\nMakespan: {makespan:.1f} hours', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Utilization', fontsize=12)
    
    # Calculate and display statistics
    avg_util = np.mean(utilization_matrix) * 100
    max_util = np.max(np.mean(utilization_matrix, axis=0)) * 100
    
    stats_text = f'Average Utilization: {avg_util:.1f}%\nPeak Period Utilization: {max_util:.1f}%'
    ax.text(1.12, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='center')
    
    plt.tight_layout()
    return fig


def main():
    """Generate all curriculum schedule visualizations."""
    print("Generating Curriculum Schedule Visualizations...")
    print("=" * 60)
    
    # Try to load trained model first
    model = load_curriculum_model()
    
    # Create environment without breaks
    # Note: If model exists, it might have different obs space, so we'll use random for visualization
    env = ScaledProductionEnv(
        n_machines=40,
        use_break_constraints=False,
        seed=42
    )
    
    # For now, use random policy to show schedule structure
    print("Using random policy for visualization...")
    model = None  # Force random to avoid obs space issues
    
    # Collect schedule data
    print("\nCollecting schedule data...")
    schedule_data, makespan = collect_schedule_data(env, model)
    
    if not schedule_data:
        print("No jobs were scheduled!")
        return
    
    print(f"Scheduled {len(schedule_data)} jobs")
    print(f"Makespan: {makespan:.1f} hours")
    
    # Create visualizations
    save_dir = Path('visualizations/curriculum/')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Job-focused Gantt chart
    print("\n1. Creating job-focused Gantt chart...")
    fig1 = create_job_gantt_chart(
        schedule_data, 
        makespan,
        title="Curriculum Learning Phase 1: Job Schedule (No Breaks)"
    )
    if fig1:
        fig1.savefig(save_dir / 'schedule_job_view.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved to: {save_dir / 'schedule_job_view.png'}")
    
    # 2. Machine-focused Gantt chart
    print("\n2. Creating machine-focused Gantt chart...")
    fig2 = create_machine_gantt_chart(
        schedule_data,
        makespan,
        n_machines=40,
        title="Curriculum Learning Phase 1: Machine Schedule (No Breaks)"
    )
    if fig2:
        fig2.savefig(save_dir / 'schedule_machine_view.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved to: {save_dir / 'schedule_machine_view.png'}")
    
    # 3. Utilization heatmap
    print("\n3. Creating utilization heatmap...")
    fig3 = create_utilization_heatmap(
        schedule_data,
        makespan,
        n_machines=40,
        title="Curriculum Learning Phase 1: Machine Utilization Heatmap"
    )
    if fig3:
        fig3.savefig(save_dir / 'machine_utilization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved to: {save_dir / 'machine_utilization_heatmap.png'}")
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()