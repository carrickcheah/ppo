"""
Visualize Real Foundation Stage Scheduling Results
Creates Job and Machine Allocation charts from actual trained model results
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal


class RealScheduleVisualizer:
    """Creates schedule visualizations from real training results."""
    
    def __init__(self):
        self.colors = {
            'on_time': '#4CAF50',      # Green
            'warning': '#FF9800',      # Orange  
            'caution': '#9C27B0',      # Purple
            'late': '#F44336',         # Red
            'unavailable': '#9E9E9E',  # Gray
            'no_action': '#E0E0E0'     # Light gray
        }
        
        self.foundation_stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
        self.checkpoint_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation"
        self.output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_schedule_from_stage(self, stage_name: str) -> Dict:
        """Run model and extract actual schedule."""
        print(f"\nProcessing {stage_name}...")
        
        # Create environment
        env = CurriculumEnvironmentReal(stage_name=stage_name, verbose=False)
        
        # Load model
        model_path = os.path.join(self.checkpoint_dir, stage_name, "final_model.zip")
        if not os.path.exists(model_path):
            print(f"No model found for {stage_name}")
            return None
        
        model = PPO.load(model_path)
        
        # Run episode
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
        
        # Extract schedule data from environment
        schedule_data = {
            'stage': stage_name,
            'job_assignments': env.job_assignments.copy(),
            'machine_schedules': env.machine_schedules.copy(),
            'families': env.families.copy(),
            'machines': [m['machine_name'] for m in env.machines],
            'machine_ids': env.machine_ids.copy(),
            'total_tasks': env.total_tasks,
            'scheduled_count': len(env.scheduled_jobs),
            'completed_count': len(env.completed_jobs),
            'scheduling_rate': len(env.scheduled_jobs) / env.total_tasks if env.total_tasks > 0 else 0
        }
        
        print(f"  - Jobs scheduled: {schedule_data['scheduled_count']}/{schedule_data['total_tasks']}")
        print(f"  - Scheduling rate: {schedule_data['scheduling_rate']:.1%}")
        
        return schedule_data
    
    def calculate_status(self, end_time: float, lcd_days: float) -> str:
        """Calculate job status based on LCD."""
        days_to_lcd = lcd_days - (end_time / 24.0)
        
        if days_to_lcd < 0:
            return 'late'
        elif days_to_lcd < 1:
            return 'warning'
        elif days_to_lcd < 3:
            return 'caution'
        else:
            return 'on_time'
    
    def create_job_allocation_chart(self, schedule: Dict, output_path: str):
        """Create Job Allocation Gantt chart."""
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Prepare job data
        all_jobs = []
        
        # Process each family
        for family_id, family_data in schedule['families'].items():
            total_sequences = family_data['total_sequences']
            lcd_days = family_data['lcd_days_remaining']
            
            # Check scheduled jobs
            for seq in range(1, total_sequences + 1):
                job_key = f"{family_id}_seq{seq}"
                
                if job_key in schedule['job_assignments']:
                    job_info = schedule['job_assignments'][job_key]
                    status = self.calculate_status(job_info['end'], lcd_days)
                    
                    # Get machine name from machine ID
                    machine_ids = job_info['machines']
                    if machine_ids and len(schedule['machines']) > 0:
                        # Map machine ID to name
                        machine_idx = schedule['machine_ids'].index(machine_ids[0])
                        machine_name = schedule['machines'][machine_idx] if machine_idx < len(schedule['machines']) else f"M{machine_ids[0]}"
                    else:
                        machine_name = f"M{machine_ids[0]}" if machine_ids else "Unknown"
                    
                    all_jobs.append({
                        'family': family_id,
                        'sequence': f"{seq}/{total_sequences}",
                        'start': job_info['start'],
                        'end': job_info['end'],
                        'machine': machine_name,
                        'status': status,
                        'scheduled': True
                    })
                else:
                    # Unscheduled job
                    all_jobs.append({
                        'family': family_id,
                        'sequence': f"{seq}/{total_sequences}",
                        'start': None,
                        'end': None,
                        'machine': 'Unscheduled',
                        'status': 'no_action',
                        'scheduled': False
                    })
        
        # Sort jobs by family and sequence
        all_jobs.sort(key=lambda x: (x['family'], int(x['sequence'].split('/')[0])))
        
        # Plot jobs
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        current_family = None
        for job in reversed(all_jobs):  # Reverse to show seq 1 at bottom
            # Add family separator
            if job['family'] != current_family:
                current_family = job['family']
                if y_pos > 0:
                    ax.axhline(y=y_pos - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            
            if job['scheduled']:
                # Plot scheduled job
                duration = job['end'] - job['start']
                ax.barh(y_pos, duration, left=job['start'], height=0.8,
                       color=self.colors[job['status']], edgecolor='black', linewidth=0.5)
            else:
                # Show unscheduled as thin gray bar
                ax.barh(y_pos, 0.5, left=23.5, height=0.8,
                       color=self.colors['no_action'], edgecolor='black', 
                       linewidth=0.5, alpha=0.5)
            
            # Label
            label = f"{job['family']}_{job['sequence']}"
            y_labels.append(label)
            y_ticks.append(y_pos)
            y_pos += 1
        
        # LCD line at 16 hours (typical deadline)
        ax.axvline(x=16, color='red', linestyle='--', linewidth=2, label='LCD')
        
        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Jobs', fontsize=12)
        ax.set_title(f'Production Planning System - {schedule["stage"].upper()}\n' +
                    f'Scheduling Rate: {schedule["scheduling_rate"]:.1%} ' +
                    f'({schedule["scheduled_count"]}/{schedule["total_tasks"]} jobs)',
                    fontsize=16, fontweight='bold')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)], rotation=45)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['late'], label='Late (<0h)'),
            mpatches.Patch(color=self.colors['warning'], label='Warning (<24h)'),
            mpatches.Patch(color=self.colors['caution'], label='Caution (<72h)'),
            mpatches.Patch(color=self.colors['on_time'], label='OK (>72h)'),
            mpatches.Patch(color=self.colors['no_action'], label='Unscheduled')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_machine_allocation_chart(self, schedule: Dict, output_path: str):
        """Create Machine Allocation Gantt chart."""
        fig, ax = plt.subplots(figsize=(20, 10))
        
        machines = schedule['machines']
        machine_schedules = schedule['machine_schedules']
        
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        # Plot each machine
        for machine_id in schedule['machine_ids']:
            # Find machine name
            machine_name = next((m for m in machines if any(
                md['machine_id'] == machine_id for md in schedule['machines'] 
                if isinstance(schedule['machines'], list) and isinstance(md, dict)
            )), f"Machine_{machine_id}")
            
            if isinstance(machines, list) and len(machines) > 0:
                if machine_id - 1 < len(machines):
                    machine_name = machines[machine_id - 1]
            
            # Get jobs on this machine
            if machine_id in machine_schedules:
                for job in machine_schedules[machine_id]:
                    # Extract family_id and sequence from job key (format: "familyID_seqN")
                    job_key = job['job']
                    parts = job_key.split('_seq')
                    if len(parts) == 2:
                        family_id = parts[0]
                        sequence = parts[1]
                    else:
                        family_id = job_key
                        sequence = '1'
                    
                    # Calculate status
                    if family_id in schedule['families']:
                        lcd_days = schedule['families'][family_id]['lcd_days_remaining']
                        status = self.calculate_status(job['end'], lcd_days)
                    else:
                        status = 'on_time'
                    
                    # Plot job
                    duration = job['end'] - job['start']
                    ax.barh(y_pos, duration, left=job['start'], height=0.8,
                           color=self.colors[status], edgecolor='black', linewidth=0.5)
                    
                    # Add job text
                    job_text = f"{family_id}_seq{sequence}"
                    bar_center = job['start'] + duration / 2
                    if duration > 1:  # Only add text if bar is wide enough
                        ax.text(bar_center, y_pos, job_text, ha='center', va='center',
                               fontsize=7, bbox=dict(boxstyle='round,pad=0.2',
                                                   facecolor='white', alpha=0.7))
            
            y_labels.append(machine_name)
            y_ticks.append(y_pos)
            y_pos += 1
        
        # LCD line
        ax.axvline(x=16, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_title(f'Machine Allocation - {schedule["stage"].upper()}\n' +
                    f'Machines: {len(machines)}, Jobs Scheduled: {schedule["scheduled_count"]}',
                    fontsize=16, fontweight='bold')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)], rotation=45)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['late'], label='Late (<0h)'),
            mpatches.Patch(color=self.colors['warning'], label='Warning (<24h)'),
            mpatches.Patch(color=self.colors['caution'], label='Caution (<72h)'),
            mpatches.Patch(color=self.colors['on_time'], label='OK (>72h)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add 3d tab highlight
        tab_y = ax.get_ylim()[1] + 0.5
        tabs = ['1d', '2d', '3d', '4d', '5d', '7d', '14d', '21d', '1m', '2m', '3m', 'all']
        tab_x_start = 2
        for i, tab in enumerate(tabs):
            if tab == '3d':
                ax.text(tab_x_start + i*1.5, tab_y, tab, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#2196F3'),
                       color='white', ha='center', fontsize=9, fontweight='bold')
            else:
                ax.text(tab_x_start + i*1.5, tab_y, tab,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'),
                       ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_charts(self):
        """Generate charts for all foundation stages."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("="*60)
        print("Generating Real Schedule Visualizations")
        print("="*60)
        
        for stage in self.foundation_stages:
            # Extract schedule
            schedule = self.extract_schedule_from_stage(stage)
            
            if schedule:
                # Generate job allocation chart
                job_path = os.path.join(self.output_dir, 
                                      f'q_{stage}_job_allocation_{timestamp}.png')
                self.create_job_allocation_chart(schedule, job_path)
                print(f"  ✓ Job allocation chart: {job_path}")
                
                # Generate machine allocation chart
                machine_path = os.path.join(self.output_dir,
                                          f'q_{stage}_machine_allocation_{timestamp}.png')
                self.create_machine_allocation_chart(schedule, machine_path)
                print(f"  ✓ Machine allocation chart: {machine_path}")
                
                print()


def main():
    visualizer = RealScheduleVisualizer()
    visualizer.generate_all_charts()
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()