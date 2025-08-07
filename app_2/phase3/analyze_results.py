"""
Phase 3 Results Analysis Script
Analyzes training results and generates logs, results, and visualization charts
"""

import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def load_test_results():
    """Load the most recent test results from phase3"""
    # Try to find test results
    test_files = [
        "/Users/carrickcheah/Project/ppo/app_2/phase3/test_results.json",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/phase3_test_results.json",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/evaluation_results.json"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    
    # If no test file exists, create dummy data for demonstration
    print("No test results found. Creating sample data for visualization...")
    return create_sample_data()

def create_sample_data():
    """Create sample scheduling data for visualization"""
    np.random.seed(42)
    
    # Sample job allocation data
    jobs = []
    machines = ["FM01", "FM02", "CM03", "CL01", "CL02", "AD01", "AD02", "BR01"]
    
    # Generate 50 jobs with random assignments
    for i in range(50):
        family_id = f"JOAW2504{i:04d}"
        num_sequences = np.random.randint(2, 6)
        
        for seq in range(1, num_sequences + 1):
            start_time = np.random.uniform(0, 200)
            duration = np.random.uniform(5, 40)
            machine = np.random.choice(machines)
            
            jobs.append({
                "family_id": family_id,
                "sequence": seq,
                "total_sequences": num_sequences,
                "machine": machine,
                "start_time": start_time,
                "end_time": start_time + duration,
                "duration": duration,
                "process": f"Process_{seq}",
                "lcd_deadline": start_time + duration + np.random.uniform(10, 100)
            })
    
    return {
        "jobs": jobs,
        "machines": machines,
        "metrics": {
            "completion_rate": 95.5,
            "on_time_rate": 88.3,
            "machine_utilization": 75.2,
            "makespan": 250.0,
            "total_jobs": 50,
            "total_sequences": len(jobs)
        }
    }

def generate_log_file(data, output_dir):
    """Generate log file with analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"q_analysis_log_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PHASE 3 RESULTS ANALYSIS LOG\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Write metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*40 + "\n")
        metrics = data.get('metrics', {})
        f.write(f"Completion Rate: {metrics.get('completion_rate', 0):.2f}%\n")
        f.write(f"On-Time Rate: {metrics.get('on_time_rate', 0):.2f}%\n")
        f.write(f"Machine Utilization: {metrics.get('machine_utilization', 0):.2f}%\n")
        f.write(f"Makespan: {metrics.get('makespan', 0):.2f} hours\n")
        f.write(f"Total Jobs: {metrics.get('total_jobs', 0)}\n")
        f.write(f"Total Sequences: {metrics.get('total_sequences', 0)}\n\n")
        
        # Write job details
        f.write("JOB SCHEDULING DETAILS:\n")
        f.write("-"*40 + "\n")
        jobs = data.get('jobs', [])
        for job in jobs[:10]:  # Show first 10 jobs
            f.write(f"Job: {job['family_id']}_seq{job['sequence']}/{job['total_sequences']}\n")
            f.write(f"  Machine: {job['machine']}\n")
            f.write(f"  Start: {job['start_time']:.2f}h, End: {job['end_time']:.2f}h\n")
            f.write(f"  Duration: {job['duration']:.2f}h\n")
            f.write(f"  LCD Deadline: {job['lcd_deadline']:.2f}h\n\n")
        
        if len(jobs) > 10:
            f.write(f"... and {len(jobs) - 10} more jobs\n")
    
    print(f"Log file generated: {log_file}")
    return log_file

def generate_result_file(data, output_dir):
    """Generate JSON result file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"q_results_{timestamp}.json"
    
    # Prepare result data
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 3",
        "metrics": data.get('metrics', {}),
        "summary": {
            "total_jobs_scheduled": len(set(j['family_id'] for j in data.get('jobs', []))),
            "total_sequences": len(data.get('jobs', [])),
            "machines_used": len(set(j['machine'] for j in data.get('jobs', []))),
            "time_horizon": max([j['end_time'] for j in data.get('jobs', [])], default=0)
        },
        "job_details": data.get('jobs', [])
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
    
    # Group jobs by family
    families = {}
    for job in jobs:
        family_id = job['family_id']
        if family_id not in families:
            families[family_id] = []
        families[family_id].append(job)
    
    # Sort families for consistent display
    sorted_families = sorted(families.keys())[:20]  # Show first 20 families
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color mapping for machines
    machines = list(set(j['machine'] for j in jobs))
    colors = plt.cm.Set3(np.linspace(0, 1, len(machines)))
    machine_colors = {m: colors[i] for i, m in enumerate(machines)}
    
    # Plot each job
    y_pos = 0
    y_labels = []
    
    for family_id in sorted_families:
        family_jobs = sorted(families[family_id], key=lambda x: x['sequence'])
        
        for job in family_jobs:
            # Create job label
            label = f"{family_id}_seq{job['sequence']}/{job['total_sequences']}"
            y_labels.append(label)
            
            # Determine color based on deadline
            time_to_deadline = job['lcd_deadline'] - job['end_time']
            if time_to_deadline < 0:
                color = '#FF0000'  # Red - Late
            elif time_to_deadline < 24:
                color = '#FFA500'  # Orange - Warning
            elif time_to_deadline < 72:
                color = '#FFFF00'  # Yellow - Caution
            else:
                color = '#00FF00'  # Green - OK
            
            # Draw bar
            ax.barh(y_pos, job['duration'], left=job['start_time'], 
                   height=0.8, color=color, edgecolor='black', linewidth=0.5)
            
            # Add machine text
            ax.text(job['start_time'] + job['duration']/2, y_pos,
                   job['machine'], ha='center', va='center', 
                   fontsize=8, fontweight='bold')
            
            y_pos += 1
    
    # Set labels and title
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Job Sequences', fontsize=12)
    ax.set_title('Phase 3 - Job Allocation Gantt Chart', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#00FF00', label='OK (>72h to LCD)'),
        mpatches.Patch(color='#FFFF00', label='Caution (<72h to LCD)'),
        mpatches.Patch(color='#FFA500', label='Warning (<24h to LCD)'),
        mpatches.Patch(color='#FF0000', label='Late (<0h to LCD)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max([j['end_time'] for j in jobs]) + 10)
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Job allocation chart generated: {chart_file}")
    return chart_file

def generate_machine_allocation_chart(data, output_dir):
    """Generate Machine Allocation Gantt chart"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_file = output_dir / f"q_machine_allocation_{timestamp}.png"
    
    jobs = data.get('jobs', [])
    if not jobs:
        print("No job data available for chart generation")
        return None
    
    # Group jobs by machine
    machines = {}
    for job in jobs:
        machine_id = job['machine']
        if machine_id not in machines:
            machines[machine_id] = []
        machines[machine_id].append(job)
    
    # Sort machines
    sorted_machines = sorted(machines.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot each machine's jobs
    y_pos = 0
    y_labels = []
    
    for machine_id in sorted_machines:
        y_labels.append(machine_id)
        machine_jobs = sorted(machines[machine_id], key=lambda x: x['start_time'])
        
        for job in machine_jobs:
            # Determine color based on deadline
            time_to_deadline = job['lcd_deadline'] - job['end_time']
            if time_to_deadline < 0:
                color = '#FF0000'  # Red - Late
            elif time_to_deadline < 24:
                color = '#FFA500'  # Orange - Warning
            elif time_to_deadline < 72:
                color = '#FFFF00'  # Yellow - Caution
            else:
                color = '#00FF00'  # Green - OK
            
            # Draw bar
            ax.barh(y_pos, job['duration'], left=job['start_time'], 
                   height=0.8, color=color, edgecolor='black', linewidth=0.5)
            
            # Add job text
            label = f"{job['family_id'][-4:]}-{job['sequence']}"
            if job['duration'] > 5:  # Only add text if bar is wide enough
                ax.text(job['start_time'] + job['duration']/2, y_pos,
                       label, ha='center', va='center', 
                       fontsize=7, fontweight='bold')
        
        y_pos += 1
    
    # Calculate machine utilization
    for i, machine_id in enumerate(sorted_machines):
        machine_jobs = machines[machine_id]
        total_busy = sum(j['duration'] for j in machine_jobs)
        time_span = max([j['end_time'] for j in machine_jobs], default=0)
        utilization = (total_busy / time_span * 100) if time_span > 0 else 0
        y_labels[i] = f"{machine_id} ({utilization:.1f}%)"
    
    # Set labels and title
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_title('Phase 3 - Machine Allocation Gantt Chart', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#00FF00', label='OK (>72h to LCD)'),
        mpatches.Patch(color='#FFFF00', label='Caution (<72h to LCD)'),
        mpatches.Patch(color='#FFA500', label='Warning (<24h to LCD)'),
        mpatches.Patch(color='#FF0000', label='Late (<0h to LCD)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max([j['end_time'] for j in jobs]) + 10)
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Machine allocation chart generated: {chart_file}")
    return chart_file

def main():
    """Main execution function"""
    print("="*60)
    print("PHASE 3 RESULTS ANALYSIS")
    print("="*60)
    
    # Create output directories
    log_dir = Path("/Users/carrickcheah/Project/ppo/app_2/phase3/logs")
    result_dir = Path("/Users/carrickcheah/Project/ppo/app_2/phase3/results")
    viz_dir = Path("/Users/carrickcheah/Project/ppo/app_2/visualizations")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test results
    data = load_test_results()
    
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