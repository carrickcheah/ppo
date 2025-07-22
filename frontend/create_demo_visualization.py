import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Define machines
machines = ['CM03', 'CL02', 'AD02-50HP', 'PP33-250T', 'OV01']
machine_y_pos = {machine: i for i, machine in enumerate(machines)}

# Define colors for different jobs
colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1',
          '#d084d0', '#ffb347', '#67b7dc', '#a4de6c', '#ffd93d']

# Create mock scheduled jobs
scheduled_jobs = []
current_times = {machine: 0 for machine in machines}

# Generate 20 jobs
for i in range(20):
    # Select machine (round-robin with some variation)
    machine_idx = i % len(machines)
    if i > 10 and i % 3 == 0:  # Add some variation
        machine_idx = (machine_idx + 1) % len(machines)
    
    machine = machines[machine_idx]
    
    # Job properties
    job_id = f"JOAW{str(i+1).zfill(4)}"
    start_time = current_times[machine]
    processing_time = 2.5 + (i % 4) * 0.5
    setup_time = 0.3
    total_time = processing_time + setup_time
    end_time = start_time + total_time
    
    scheduled_jobs.append({
        'job_id': job_id,
        'machine': machine,
        'start_time': start_time,
        'end_time': end_time,
        'color': colors[i % len(colors)]
    })
    
    current_times[machine] = end_time

# Plot jobs as rectangles
for job in scheduled_jobs:
    y_pos = machine_y_pos[job['machine']]
    rect = patches.Rectangle(
        (job['start_time'], y_pos - 0.4),
        job['end_time'] - job['start_time'],
        0.8,
        linewidth=1,
        edgecolor='black',
        facecolor=job['color'],
        alpha=0.8
    )
    ax.add_patch(rect)
    
    # Add job ID text if rectangle is wide enough
    if job['end_time'] - job['start_time'] > 1:
        ax.text(
            (job['start_time'] + job['end_time']) / 2,
            y_pos,
            job['job_id'][-4:],
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='white'
        )

# Calculate metrics
makespan = max(job['end_time'] for job in scheduled_jobs)
total_machine_time = sum(job['end_time'] - job['start_time'] for job in scheduled_jobs)
total_available_time = makespan * len(machines)
utilization = (total_machine_time / total_available_time) * 100

# Set axis labels and title
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Machines', fontsize=12)
ax.set_title('PPO Production Schedule - Gantt Chart Visualization', fontsize=16, fontweight='bold')

# Set y-axis
ax.set_yticks(range(len(machines)))
ax.set_yticklabels(machines)

# Set x-axis
ax.set_xlim(0, makespan * 1.05)
ax.set_xticks(np.arange(0, makespan + 5, 5))

# Add grid
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

# Add makespan line
ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(makespan + 0.5, len(machines) - 0.5, f'Makespan: {makespan:.1f}h', 
        rotation=0, fontsize=10, color='red', fontweight='bold')

# Add metrics box
metrics_text = f"Schedule Metrics:\n" \
               f"Total Jobs: 20\n" \
               f"Makespan: {makespan:.1f} hours\n" \
               f"Machine Utilization: {utilization:.1f}%\n" \
               f"Completion Rate: 100%"

props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Adjust layout
plt.tight_layout()

# Save the visualization
output_path = '/Users/carrickcheah/Project/ppo/frontend/demo_gantt_chart.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Gantt chart saved to: {output_path}")

# Also create a simpler version showing the frontend interface
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})

# Top panel - metrics cards
metrics = [
    ('Makespan', f'{makespan:.1f} hours', '#1976d2'),
    ('Completion Rate', '100%', '#4caf50'),
    ('Machine Utilization', f'{utilization:.1f}%', '#9c27b0'),
    ('Jobs On Time', '95%', '#4caf50')
]

ax1.set_xlim(0, 4)
ax1.set_ylim(0, 1)
ax1.axis('off')

for i, (label, value, color) in enumerate(metrics):
    x = i + 0.5
    # Card background
    rect = patches.FancyBboxPatch((x-0.4, 0.2), 0.8, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor='white',
                                  edgecolor=color,
                                  linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x, 0.65, label, ha='center', fontsize=10, color='gray')
    ax1.text(x, 0.35, value, ha='center', fontsize=14, fontweight='bold', color=color)

ax1.text(2, 0.95, 'PPO Production Scheduler Dashboard', ha='center', fontsize=18, fontweight='bold')

# Bottom panel - simplified gantt
ax2.set_xlabel('Time (hours)', fontsize=12)
ax2.set_ylabel('Machines', fontsize=12)
ax2.set_title('Schedule Visualization', fontsize=14)

# Plot simplified schedule
for job in scheduled_jobs[:10]:  # Show only first 10 jobs for clarity
    y_pos = machine_y_pos[job['machine']]
    rect = patches.Rectangle(
        (job['start_time'], y_pos - 0.35),
        job['end_time'] - job['start_time'],
        0.7,
        linewidth=1,
        edgecolor='black',
        facecolor=job['color'],
        alpha=0.8
    )
    ax2.add_patch(rect)
    
    ax2.text(
        (job['start_time'] + job['end_time']) / 2,
        y_pos,
        job['job_id'][-4:],
        ha='center',
        va='center',
        fontsize=8,
        color='white'
    )

ax2.set_yticks(range(len(machines)))
ax2.set_yticklabels(machines)
ax2.set_xlim(0, 15)
ax2.grid(True, axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
output_path2 = '/Users/carrickcheah/Project/ppo/frontend/demo_dashboard.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Dashboard demo saved to: {output_path2}")

plt.close('all')