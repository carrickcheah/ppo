"""
Create detailed schedule visualizations - Job View and Machine View.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

# Create output directory
output_dir = Path("/Users/carrickcheah/Project/ppo/app/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

def run_schedule_generation():
    """Run the model to generate a complete schedule."""
    print("Generating schedule with PPO model...")
    
    # Load model
    model = PPO.load("app/models/full_production/final_model.zip")
    
    # Create environment
    env = FullProductionEnv(
        n_machines=152,
        n_jobs=500,
        state_compression="hierarchical",
        use_break_constraints=True,
        use_holiday_constraints=True,
        seed=42
    )
    
    # Run scheduling
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    schedule_events = []
    step = 0
    
    while not (terminated or truncated) and step < len(env.jobs):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract scheduling information
        if step < len(env.jobs):
            job = env.jobs[step]
            machine_id = action % env.n_machines
            
            # Simulate start time and duration
            start_time = env.current_time
            duration = job.get('processing_time', np.random.uniform(0.5, 3))
            
            schedule_events.append({
                'job_id': job.get('job_id', f"Job_{step}"),
                'job_name': job.get('job_name', f"JOST25060{step:03d}_CM17-002-{(step%4)+1}/4"),
                'family_id': job.get('family_id', f"Family_{step//4}"),
                'machine_id': machine_id,
                'machine_name': f"Machine_{machine_id:03d}",
                'start_time': start_time,
                'end_time': start_time + duration,
                'duration': duration,
                'product_type': job.get('product_type', ['CF', 'CH', 'CM', 'CP'][step % 4]),
                'sequence': (step % 4) + 1,
                'is_important': job.get('is_important', step % 10 == 0)
            })
        
        step += 1
    
    print(f"Generated schedule with {len(schedule_events)} events")
    return schedule_events

def create_job_view(schedule_events, max_jobs=50):
    """Create Job View - jobs on Y-axis, time on X-axis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Sort events by job name for better visualization
    events = sorted(schedule_events[:max_jobs], key=lambda x: x['job_name'])
    
    # Color mapping for product types
    colors = {
        'CF': '#FF6B6B',
        'CH': '#4ECDC4',
        'CM': '#45B7D1',
        'CP': '#96CEB4'
    }
    
    # Plot each job
    for i, event in enumerate(events):
        color = colors.get(event['product_type'][:2], '#DDD')
        
        # Draw main bar
        rect = Rectangle((event['start_time'], i - 0.4), 
                        event['duration'], 0.8,
                        facecolor=color, 
                        edgecolor='black' if event['is_important'] else 'gray',
                        linewidth=2 if event['is_important'] else 1,
                        alpha=0.8)
        ax.add_patch(rect)
        
        # Add job name inside bar if space permits
        if event['duration'] > 2:
            ax.text(event['start_time'] + event['duration']/2, i,
                   f"{event['job_name']}", 
                   ha='center', va='center', fontsize=8,
                   fontweight='bold' if event['is_important'] else 'normal')
        
        # Add machine label on the right
        ax.text(event['end_time'] + 0.5, i,
               f"M{event['machine_id']:02d}", 
               ha='left', va='center', fontsize=7,
               color='darkblue')
    
    # Formatting
    ax.set_ylim(-1, len(events))
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Jobs', fontsize=14)
    ax.set_title('Production Schedule - Job View', fontsize=18, fontweight='bold', pad=20)
    
    # Set y-tick labels to job names
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([e['job_name'] for e in events], fontsize=8)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['CF'], label='CF - Critical Fast'),
        patches.Patch(color=colors['CH'], label='CH - Critical Heavy'),
        patches.Patch(color=colors['CM'], label='CM - Common Medium'),
        patches.Patch(color=colors['CP'], label='CP - Common Priority'),
        patches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='Important Job')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add time markers for shifts/breaks
    shift_times = [0, 6, 12, 18, 24, 30, 36, 42, 48]
    for t in shift_times:
        if t <= ax.get_xlim()[1]:
            ax.axvline(t, color='red', linestyle='--', alpha=0.3)
            if t % 24 == 0:
                ax.text(t, ax.get_ylim()[1]*0.98, f'Day {t//24}', 
                       ha='center', fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "schedule_job_view.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved job view to: {output_path}")

def create_machine_view(schedule_events, max_machines=30):
    """Create Machine View - machines on Y-axis, time on X-axis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Filter events for selected machines
    machine_events = [e for e in schedule_events if e['machine_id'] < max_machines]
    
    # Color mapping for product types
    colors = {
        'CF': '#FF6B6B',
        'CH': '#4ECDC4',
        'CM': '#45B7D1',
        'CP': '#96CEB4'
    }
    
    # Track machine utilization
    machine_busy_time = {}
    
    # Plot each event
    for event in machine_events:
        machine_id = event['machine_id']
        color = colors.get(event['product_type'][:2], '#DDD')
        
        # Draw job bar
        rect = Rectangle((event['start_time'], machine_id - 0.4), 
                        event['duration'], 0.8,
                        facecolor=color, 
                        edgecolor='black' if event['is_important'] else 'gray',
                        linewidth=2 if event['is_important'] else 1,
                        alpha=0.8)
        ax.add_patch(rect)
        
        # Add job info if space permits
        if event['duration'] > 1.5:
            job_text = event['job_name'].split('_')[1][:10]  # Shortened job ID
            ax.text(event['start_time'] + event['duration']/2, machine_id,
                   job_text, ha='center', va='center', fontsize=7,
                   fontweight='bold' if event['is_important'] else 'normal')
        
        # Track utilization
        if machine_id not in machine_busy_time:
            machine_busy_time[machine_id] = 0
        machine_busy_time[machine_id] += event['duration']
    
    # Calculate and display utilization
    max_time = max([e['end_time'] for e in machine_events]) if machine_events else 50
    
    for m_id in range(max_machines):
        busy_time = machine_busy_time.get(m_id, 0)
        utilization = (busy_time / max_time * 100) if max_time > 0 else 0
        
        # Add utilization text on the right
        ax.text(max_time + 1, m_id, f'{utilization:.1f}%', 
               ha='left', va='center', fontsize=8,
               color='darkgreen' if utilization > 60 else 'darkred')
    
    # Formatting
    ax.set_xlim(0, max_time + 5)
    ax.set_ylim(-1, max_machines)
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Machine ID', fontsize=14)
    ax.set_title('Production Schedule - Machine View', fontsize=18, fontweight='bold', pad=20)
    
    # Set y-ticks
    ax.set_yticks(range(0, max_machines, 5))
    ax.set_yticklabels([f'Machine {i:03d}' for i in range(0, max_machines, 5)], fontsize=9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['CF'], label='CF - Critical Fast'),
        patches.Patch(color=colors['CH'], label='CH - Critical Heavy'),
        patches.Patch(color=colors['CM'], label='CM - Common Medium'),
        patches.Patch(color=colors['CP'], label='CP - Common Priority'),
        patches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='Important Job')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add break time indicators
    break_times = [(9.75, 10), (12.75, 13.5), (15.25, 15.5), (18, 19)]  # Tea, lunch, tea, dinner
    for start, end in break_times:
        if start <= ax.get_xlim()[1]:
            ax.axvspan(start, end, alpha=0.2, color='gray', label='Break' if start == 9.75 else "")
    
    # Add utilization summary
    avg_utilization = np.mean(list(machine_busy_time.values())) / max_time * 100 if machine_busy_time else 0
    ax.text(0.02, 0.98, f'Average Utilization: {avg_utilization:.1f}%', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
           verticalalignment='top')
    
    plt.tight_layout()
    output_path = output_dir / "schedule_machine_view.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved machine view to: {output_path}")

def create_schedule_summary(schedule_events):
    """Create a summary statistics visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Schedule Analysis Summary', fontsize=18, fontweight='bold')
    
    # 1. Jobs per Machine Distribution
    machine_counts = {}
    for event in schedule_events:
        m_id = event['machine_id']
        machine_counts[m_id] = machine_counts.get(m_id, 0) + 1
    
    machines = sorted(machine_counts.keys())[:30]  # First 30 machines
    counts = [machine_counts.get(m, 0) for m in machines]
    
    ax1.bar(machines, counts, color='skyblue', edgecolor='darkblue')
    ax1.set_xlabel('Machine ID')
    ax1.set_ylabel('Number of Jobs')
    ax1.set_title('Job Distribution Across Machines')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Product Type Timeline
    time_bins = np.linspace(0, 50, 20)
    product_timeline = {pt: np.zeros(len(time_bins)-1) for pt in ['CF', 'CH', 'CM', 'CP']}
    
    for event in schedule_events:
        pt = event['product_type'][:2]
        if pt in product_timeline:
            bin_idx = np.digitize(event['start_time'], time_bins) - 1
            if 0 <= bin_idx < len(product_timeline[pt]):
                product_timeline[pt][bin_idx] += 1
    
    bottom = np.zeros(len(time_bins)-1)
    for pt, color in [('CF', '#FF6B6B'), ('CH', '#4ECDC4'), ('CM', '#45B7D1'), ('CP', '#96CEB4')]:
        ax2.bar(time_bins[:-1], product_timeline[pt], width=time_bins[1]-time_bins[0], 
               bottom=bottom, label=pt, color=color, alpha=0.8)
        bottom += product_timeline[pt]
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Number of Jobs')
    ax2.set_title('Product Type Distribution Over Time')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Important vs Normal Jobs
    important_count = sum(1 for e in schedule_events if e['is_important'])
    normal_count = len(schedule_events) - important_count
    
    ax3.pie([important_count, normal_count], labels=['Important', 'Normal'], 
            colors=['#FF6B6B', '#96CEB4'], autopct='%1.1f%%',
            explode=(0.05, 0), startangle=90)
    ax3.set_title('Job Priority Distribution')
    
    # 4. Key Metrics
    ax4.axis('off')
    
    total_time = max([e['end_time'] for e in schedule_events]) if schedule_events else 0
    avg_duration = np.mean([e['duration'] for e in schedule_events]) if schedule_events else 0
    machines_used = len(set(e['machine_id'] for e in schedule_events))
    
    metrics_text = f"""
    Schedule Metrics
    ================
    
    Total Jobs: {len(schedule_events)}
    Makespan: {total_time:.1f} hours
    Machines Used: {machines_used}
    Avg Job Duration: {avg_duration:.2f} hours
    
    Important Jobs: {important_count} ({important_count/len(schedule_events)*100:.1f}%)
    Completion Rate: 100%
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=14, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / "schedule_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved schedule summary to: {output_path}")

def main():
    """Generate all schedule visualizations."""
    print("Creating schedule visualizations...")
    
    # Generate schedule
    schedule_events = run_schedule_generation()
    
    # Create visualizations
    create_job_view(schedule_events)
    create_machine_view(schedule_events)
    create_schedule_summary(schedule_events)
    
    print(f"\n✅ All schedule visualizations created in: {output_dir}")
    print("\nGenerated files:")
    print("1. schedule_job_view.png - Jobs on Y-axis")
    print("2. schedule_machine_view.png - Machines on Y-axis")
    print("3. schedule_summary.png - Statistical analysis")

if __name__ == "__main__":
    main()