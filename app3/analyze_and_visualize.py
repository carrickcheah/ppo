"""
App3 Results Analysis and Visualization Script
Fetches data from API and generates logs, results, and visualization charts
"""

import json
import os
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def fetch_schedule_data(dataset="100_jobs", model="sb3_1million"):
    """Fetch scheduling data from the API"""
    url = "http://localhost:8000/api/schedule"
    payload = {
        "dataset": dataset,
        "model": model,
        "deterministic": True,
        "max_steps": 10000
    }
    
    try:
        print(f"Fetching schedule data for {dataset} using {model}...")
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully fetched data: {len(data.get('jobs', []))} jobs scheduled")
            return data
        else:
            print(f"Error: API returned status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return None

def generate_log_file(data, output_dir):
    """Generate log file with analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"q_analysis_log_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("APP3 PHASE 3 RESULTS ANALYSIS LOG\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Write dataset and model info
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Dataset Used: {data.get('dataset_used', 'Unknown')}\n")
        f.write(f"Model Used: {data.get('model_used', 'Unknown')}\n")
        f.write(f"Success: {data.get('success', False)}\n\n")
        
        # Write metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*40 + "\n")
        stats = data.get('statistics', {})
        f.write(f"Completion Rate: {stats.get('completion_rate', 0):.2f}%\n")
        f.write(f"On-Time Rate: {stats.get('on_time_rate', 0):.2f}%\n")
        f.write(f"Machine Utilization: {stats.get('machine_utilization', 0):.2f}%\n")
        f.write(f"Makespan: {stats.get('makespan', 0):.2f} hours\n")
        f.write(f"Scheduled Tasks: {stats.get('scheduled_tasks', 0)}/{stats.get('total_tasks', 0)}\n")
        f.write(f"On-Time Tasks: {stats.get('on_time_tasks', 0)}\n")
        f.write(f"Late Tasks: {stats.get('late_tasks', 0)}\n")
        f.write(f"Average Tardiness: {stats.get('average_tardiness', 0):.2f} hours\n")
        f.write(f"Total Reward: {stats.get('total_reward', 0):.2f}\n")
        f.write(f"Inference Time: {stats.get('inference_time', 0):.2f} seconds\n\n")
        
        # Write job details (first 20)
        f.write("JOB SCHEDULING DETAILS (First 20 jobs):\n")
        f.write("-"*40 + "\n")
        jobs = data.get('jobs', [])
        for job in jobs[:20]:
            f.write(f"Job: {job['task_label']}\n")
            f.write(f"  Family ID: {job['job_id']}\n")
            f.write(f"  Machine: {job['machine']}\n")
            f.write(f"  Process: {job['process_name']}\n")
            f.write(f"  Start: {job['start']:.2f}h, End: {job['end']:.2f}h\n")
            f.write(f"  Duration: {job['duration']:.2f}h\n")
            f.write(f"  LCD Deadline: {job['lcd_hours']:.2f}h\n")
            f.write(f"  Days to Deadline: {job['days_to_deadline']:.2f} days\n")
            f.write(f"  Status Color: {job['color']}\n\n")
        
        if len(jobs) > 20:
            f.write(f"... and {len(jobs) - 20} more jobs\n\n")
        
        # Write machine allocation summary
        f.write("MACHINE ALLOCATION SUMMARY:\n")
        f.write("-"*40 + "\n")
        machines = data.get('machines', [])
        for machine in machines[:10]:
            f.write(f"Machine: {machine['machine_name']} (ID: {machine['machine_id']})\n")
            f.write(f"  Tasks Assigned: {len(machine['tasks'])}\n")
            f.write(f"  Utilization: {machine['utilization']:.2f}%\n")
            f.write(f"  Total Busy Time: {machine['total_busy_time']:.2f} hours\n\n")
    
    print(f"Log file generated: {log_file}")
    return log_file

def generate_result_file(data, output_dir):
    """Generate JSON result file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"q_results_{timestamp}.json"
    
    # Prepare result data
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "App3 Phase 3",
        "configuration": {
            "dataset": data.get('dataset_used', 'Unknown'),
            "model": data.get('model_used', 'Unknown'),
            "success": data.get('success', False)
        },
        "metrics": data.get('statistics', {}),
        "summary": {
            "total_jobs_scheduled": len(set(j['job_id'] for j in data.get('jobs', []))),
            "total_sequences": len(data.get('jobs', [])),
            "machines_used": len(data.get('machines', [])),
            "time_horizon": data.get('statistics', {}).get('makespan', 0)
        },
        "job_details": data.get('jobs', []),
        "machine_details": data.get('machines', [])
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Result file generated: {result_file}")
    return result_file

def generate_job_allocation_chart(data, output_dir):
    """Generate Jobs Allocation Gantt chart"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = output_dir / f"q_job_allocation_{timestamp}.png"
    
    jobs = data.get('jobs', [])
    if not jobs:
        print("No job data available for chart generation")
        return None
    
    # Sort jobs by family and sequence
    jobs_sorted = sorted(jobs, key=lambda x: (x['job_id'], x['sequence']))
    
    # Take first 50 jobs for visualization
    jobs_to_plot = jobs_sorted[:50]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Plot each job
    y_pos = 0
    y_labels = []
    
    for job in jobs_to_plot:
        # Create job label
        y_labels.append(job['task_label'])
        
        # Use the color from API (based on deadline)
        color = job['color']
        
        # Draw bar
        ax.barh(y_pos, job['duration'], left=job['start'], 
               height=0.8, color=color, edgecolor='black', linewidth=0.5)
        
        # Add machine text
        ax.text(job['start'] + job['duration']/2, y_pos,
               job['machine'], ha='center', va='center', 
               fontsize=8, fontweight='bold', color='black')
        
        y_pos += 1
    
    # Set labels and title
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Job Sequences', fontsize=12, fontweight='bold')
    ax.set_title(f'App3 Phase 3 - Job Allocation Gantt Chart\n{data.get("dataset_used", "")} with {data.get("model_used", "")}', 
                fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#00FF00', label='OK (>72h to LCD)'),
        mpatches.Patch(color='#FFFF00', label='Caution (<72h to LCD)'),
        mpatches.Patch(color='#FFA500', label='Warning (<24h to LCD)'),
        mpatches.Patch(color='#FF0000', label='Late (<0h to LCD)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max([j['end'] for j in jobs_to_plot]) + 10)
    
    # Add performance metrics as text
    stats = data.get('statistics', {})
    metrics_text = f"Completion: {stats.get('completion_rate', 0):.1f}% | On-Time: {stats.get('on_time_rate', 0):.1f}% | Utilization: {stats.get('machine_utilization', 0):.1f}%"
    ax.text(0.5, -0.08, metrics_text, transform=ax.transAxes, 
           ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Job allocation chart generated: {chart_file}")
    return chart_file

def generate_machine_allocation_chart(data, output_dir):
    """Generate Machine Allocation Gantt chart"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = output_dir / f"q_machine_allocation_{timestamp}.png"
    
    machines = data.get('machines', [])
    if not machines:
        print("No machine data available for chart generation")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plot each machine's jobs
    y_pos = 0
    y_labels = []
    
    for machine in machines:
        # Create machine label with utilization
        label = f"{machine['machine_name']} ({machine['utilization']:.1f}%)"
        y_labels.append(label)
        
        # Plot all tasks on this machine
        for task in machine['tasks']:
            # Use the color from task (based on deadline)
            color = task['color']
            
            # Draw bar
            ax.barh(y_pos, task['duration'], left=task['start'], 
                   height=0.8, color=color, edgecolor='black', linewidth=0.5)
            
            # Add job text (shortened for readability)
            if task['duration'] > 5:  # Only add text if bar is wide enough
                job_label = task['job_id'][-8:] + f"-{task['sequence']}"
                ax.text(task['start'] + task['duration']/2, y_pos,
                       job_label, ha='center', va='center', 
                       fontsize=7, fontweight='bold', color='black')
        
        y_pos += 1
    
    # Set labels and title
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Machines', fontsize=12, fontweight='bold')
    ax.set_title(f'App3 Phase 3 - Machine Allocation Gantt Chart\n{data.get("dataset_used", "")} with {data.get("model_used", "")}', 
                fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#00FF00', label='OK (>72h to LCD)'),
        mpatches.Patch(color='#FFFF00', label='Caution (<72h to LCD)'),
        mpatches.Patch(color='#FFA500', label='Warning (<24h to LCD)'),
        mpatches.Patch(color='#FF0000', label='Late (<0h to LCD)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Set x-axis limit
    max_time = max([task['end'] for machine in machines for task in machine['tasks']], default=100)
    ax.set_xlim(0, max_time + 10)
    
    # Add performance metrics as text
    stats = data.get('statistics', {})
    metrics_text = f"Avg Utilization: {stats.get('machine_utilization', 0):.1f}% | Makespan: {stats.get('makespan', 0):.0f}h | Total Machines: {len(machines)}"
    ax.text(0.5, -0.08, metrics_text, transform=ax.transAxes, 
           ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Machine allocation chart generated: {chart_file}")
    return chart_file

def main():
    """Main execution function"""
    print("="*60)
    print("APP3 PHASE 3 RESULTS ANALYSIS")
    print("="*60)
    
    # Create output directories
    log_dir = Path("/Users/carrickcheah/Project/ppo/app3/phase3/logs")
    result_dir = Path("/Users/carrickcheah/Project/ppo/app3/phase3/results")
    viz_dir = Path("/Users/carrickcheah/Project/ppo/app3/visualizations")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch schedule data from API
    data = fetch_schedule_data(dataset="100_jobs", model="sb3_1million")
    
    if not data:
        print("Failed to fetch data from API. Exiting...")
        return
    
    # Generate outputs
    print("\nGenerating outputs...")
    print("-"*40)
    
    # 1. Generate log file
    log_file = generate_log_file(data, log_dir)
    
    # 2. Generate result file
    result_file = generate_result_file(data, result_dir)
    
    # 3. Generate job allocation chart
    job_chart = generate_job_allocation_chart(data, viz_dir)
    
    # 4. Generate machine allocation chart
    machine_chart = generate_machine_allocation_chart(data, viz_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Log file: {log_file}")
    print(f"Result file: {result_file}")
    print(f"Job allocation chart: {job_chart}")
    print(f"Machine allocation chart: {machine_chart}")

if __name__ == "__main__":
    main()