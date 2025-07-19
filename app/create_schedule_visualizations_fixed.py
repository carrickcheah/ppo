"""
Create detailed schedule visualizations with proper data extraction from environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

# Create output directory
output_dir = Path("/Users/carrickcheah/Project/ppo/app/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

def run_and_extract_schedule():
    """Run the model and extract actual scheduling data."""
    print("Running PPO model to generate schedule...")
    
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
    
    print(f"Starting scheduling for {len(env.jobs)} jobs...")
    
    while not (terminated or truncated) and step < len(env.jobs):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract scheduling information from info dict
        if 'scheduled_job' in info:
            # Parse the scheduling info
            schedule_events.append({
                'job_id': info['scheduled_job'],
                'machine_name': info['on_machine'],
                'machine_id': int(info['on_machine'].split()[0]) - 1,  # Extract machine ID
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'duration': info['processing_time'],
                'setup_time': info['setup_time'],
                'is_important': info['is_important'],
                'machine_type': info['machine_type']
            })
            
            if step % 20 == 0:
                print(f"Step {step}: Scheduled {info['scheduled_job']} on {info['on_machine']}")
        
        step += 1
    
    print(f"\nCompleted! Generated {len(schedule_events)} scheduling events")
    print(f"Final makespan: {env.current_time:.1f} hours")
    
    return schedule_events, env

def create_job_view(schedule_events, max_jobs=50):
    """Create Job View visualization."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Take first max_jobs events
    events = schedule_events[:max_jobs]
    
    # Color based on importance
    colors = ['#FF6B6B', '#4ECDC4']  # Red for important, Teal for normal
    
    # Plot each job
    for i, event in enumerate(events):
        color = colors[0] if event['is_important'] else colors[1]
        
        # Draw main bar
        rect = Rectangle((event['start_time'], i - 0.4), 
                        event['duration'], 0.8,
                        facecolor=color, 
                        edgecolor='black',
                        linewidth=2 if event['is_important'] else 1,
                        alpha=0.8)
        ax.add_patch(rect)
        
        # Add job ID
        ax.text(event['start_time'] + event['duration']/2, i,
               event['job_id'], 
               ha='center', va='center', fontsize=8,
               fontweight='bold' if event['is_important'] else 'normal')
        
        # Add machine name on the right
        ax.text(event['end_time'] + 0.2, i,
               event['machine_name'], 
               ha='left', va='center', fontsize=7,
               color='darkblue', alpha=0.7)
    
    # Formatting
    ax.set_ylim(-1, len(events))
    ax.set_xlim(0, max([e['end_time'] for e in events]) + 2)
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Job Number', fontsize=14)
    ax.set_title(f'PPO Production Schedule - Job View (First {len(events)} Jobs)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set y-ticks
    ax.set_yticks(range(0, len(events), 5))
    ax.set_yticklabels([f'Job {i+1}' for i in range(0, len(events), 5)])
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors[0], label='Important Jobs'),
        patches.Patch(color=colors[1], label='Normal Jobs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add break time shading
    break_times = [(1, 6.5), (9.75, 10), (12.75, 13.5), (15.25, 15.5), (18, 19), (23, 23.5)]
    for start, end in break_times:
        if start <= ax.get_xlim()[1]:
            ax.axvspan(start, end, alpha=0.1, color='gray')
    
    plt.tight_layout()
    output_path = output_dir / "schedule_job_view_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved job view to: {output_path}")

def create_machine_view(schedule_events, max_machines=30):
    """Create Machine View visualization."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Filter events for first machines
    machine_events = [e for e in schedule_events if e['machine_id'] < max_machines]
    
    # Color based on importance
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Track machine utilization
    machine_busy_time = {}
    machine_job_count = {}
    
    # Plot each event
    for event in machine_events:
        machine_id = event['machine_id']
        color = colors[0] if event['is_important'] else colors[1]
        
        # Draw job bar including setup time
        total_start = event['start_time'] - event['setup_time']
        total_duration = event['duration'] + event['setup_time']
        
        # Setup time (darker shade)
        if event['setup_time'] > 0:
            setup_rect = Rectangle((total_start, machine_id - 0.4), 
                                 event['setup_time'], 0.8,
                                 facecolor='darkgray', 
                                 edgecolor='black',
                                 linewidth=1,
                                 alpha=0.5)
            ax.add_patch(setup_rect)
        
        # Processing time
        rect = Rectangle((event['start_time'], machine_id - 0.4), 
                        event['duration'], 0.8,
                        facecolor=color, 
                        edgecolor='black',
                        linewidth=2 if event['is_important'] else 1,
                        alpha=0.8)
        ax.add_patch(rect)
        
        # Add job ID if space permits
        if event['duration'] > 0.5:
            ax.text(event['start_time'] + event['duration']/2, machine_id,
                   event['job_id'], ha='center', va='center', fontsize=6)
        
        # Track utilization
        if machine_id not in machine_busy_time:
            machine_busy_time[machine_id] = 0
            machine_job_count[machine_id] = 0
        machine_busy_time[machine_id] += total_duration
        machine_job_count[machine_id] += 1
    
    # Calculate max time
    max_time = max([e['end_time'] for e in schedule_events]) if schedule_events else 50
    
    # Add utilization info
    for m_id in range(max_machines):
        busy_time = machine_busy_time.get(m_id, 0)
        job_count = machine_job_count.get(m_id, 0)
        utilization = (busy_time / max_time * 100) if max_time > 0 else 0
        
        # Add text on the right
        text = f'{job_count} jobs\n{utilization:.1f}%'
        ax.text(max_time + 0.5, m_id, text, 
               ha='left', va='center', fontsize=8,
               color='darkgreen' if utilization > 50 else 'darkred')
    
    # Formatting
    ax.set_xlim(0, max_time + 5)
    ax.set_ylim(-1, max_machines)
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Machine ID', fontsize=14)
    ax.set_title(f'PPO Production Schedule - Machine View (First {max_machines} Machines)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Set y-ticks
    ax.set_yticks(range(0, max_machines, 5))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors[0], label='Important Jobs'),
        patches.Patch(color=colors[1], label='Normal Jobs'),
        patches.Patch(color='darkgray', label='Setup Time')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add break time shading
    break_times = [(1, 6.5), (9.75, 10), (12.75, 13.5), (15.25, 15.5), (18, 19), (23, 23.5)]
    for start, end in break_times:
        if start <= ax.get_xlim()[1]:
            ax.axvspan(start, end, alpha=0.1, color='gray')
    
    # Add summary stats
    total_jobs = len(schedule_events)
    active_machines = len(machine_busy_time)
    avg_utilization = np.mean(list(machine_busy_time.values())) / max_time * 100 if machine_busy_time else 0
    
    stats_text = f'Total Jobs: {total_jobs} | Active Machines: {active_machines} | Avg Utilization: {avg_utilization:.1f}%'
    ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, 
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / "schedule_machine_view_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved machine view to: {output_path}")

def main():
    """Generate schedule visualizations."""
    print("=== Creating PPO Schedule Visualizations ===")
    
    # Run model and extract schedule
    schedule_events, env = run_and_extract_schedule()
    
    if not schedule_events:
        print("ERROR: No scheduling events captured!")
        return
    
    # Create visualizations
    create_job_view(schedule_events)
    create_machine_view(schedule_events)
    
    print(f"\n✅ Visualizations created successfully!")
    print(f"Files saved in: {output_dir}")
    print("- schedule_job_view_fixed.png")
    print("- schedule_machine_view_fixed.png")

if __name__ == "__main__":
    main()