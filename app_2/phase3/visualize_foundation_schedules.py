"""
Visualize Foundation Stage Scheduling Results
Creates Job Allocation and Machine Allocation charts for all 4 foundation stages
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal


class FoundationScheduleVisualizer:
    """Creates production schedule visualizations for foundation stages."""
    
    def __init__(self):
        """Initialize visualizer."""
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
    
    def get_schedule_from_model(self, stage_name: str) -> Dict:
        """Run trained model and extract scheduling results."""
        print(f"\nGenerating schedule for {stage_name}...")
        
        # Create environment
        base_env = CurriculumEnvironmentReal(stage_name=stage_name, verbose=False)
        env = Monitor(base_env)
        
        # Load trained model
        model_path = os.path.join(self.checkpoint_dir, stage_name, "final_model.zip")
        if not os.path.exists(model_path):
            print(f"Warning: No model found for {stage_name}")
            return None
        
        model = PPO.load(model_path)
        
        # Run one episode to get schedule
        obs, _ = env.reset()
        done = False
        scheduled_jobs = []
        machine_schedules = {m['machine_name']: [] for m in base_env.machines}
        
        step = 0
        while not done and step < 200:  # Limit steps
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # Record scheduled job
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                # Extract job information from info
                job_info = {
                    'family_id': info.get('selected_family', ''),
                    'sequence_number': info.get('selected_sequence', 1),
                    'total_sequences': info.get('total_sequences', 1),
                    'machine_id': info.get('selected_machine_id', -1),
                    'machine_name': info.get('selected_machine_name', 'Unknown'),
                    'start_time': info.get('schedule_start', 0),
                    'end_time': info.get('schedule_end', 0),
                    'processing_time': info.get('processing_time', 0),
                    'lcd': info.get('lcd', 16)
                }
                scheduled_jobs.append(job_info)
                
                # Add to machine schedule
                if job_info['machine_name'] in machine_schedules:
                    machine_schedules[job_info['machine_name']].append(job_info)
            
            step += 1
            done = done or truncated
        
        # Get final schedule from environment
        schedule = {
            'stage': stage_name,
            'jobs': scheduled_jobs,
            'machines': machine_schedules,
            'families': base_env.families,
            'total_jobs': base_env.total_tasks,
            'jobs_scheduled': len(base_env.scheduled_jobs),
            'scheduling_rate': len(base_env.scheduled_jobs) / base_env.total_tasks if base_env.total_tasks > 0 else 0
        }
        
        return schedule
    
    def calculate_job_status(self, end_time: datetime, lcd: datetime) -> str:
        """Calculate job status based on LCD."""
        time_diff = (lcd - end_time).total_seconds() / 3600  # hours
        
        if time_diff < 0:
            return 'late'
        elif time_diff < 24:
            return 'warning'
        elif time_diff < 72:
            return 'caution'
        else:
            return 'on_time'
    
    def create_job_allocation_chart(self, schedule: Dict, output_path: str):
        """Create Job Allocation Gantt chart for a stage."""
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Organize jobs by family
        families = {}
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Process scheduled jobs
        for job in schedule['jobs']:
            family_id = job['family_id']
            if family_id not in families:
                families[family_id] = []
            
            # Calculate times
            start_time = base_time + timedelta(hours=job['start_time'])
            end_time = base_time + timedelta(hours=job['end_time'])
            lcd = base_time + timedelta(hours=job.get('lcd', 16))
            
            job_data = {
                'family': family_id,
                'sequence': f"{job['sequence_number']}/{job['total_sequences']}",
                'start': start_time,
                'end': end_time,
                'lcd': lcd,
                'status': self.calculate_job_status(end_time, lcd),
                'machine': job['machine_name']
            }
            families[family_id].append(job_data)
        
        # Add unscheduled jobs
        for family_id, family_data in schedule['families'].items():
            scheduled_sequences = [j['sequence_number'] for j in schedule['jobs'] if j['family_id'] == family_id]
            
            for seq in range(1, family_data['total_sequences'] + 1):
                if seq not in scheduled_sequences:
                    if family_id not in families:
                        families[family_id] = []
                    
                    families[family_id].append({
                        'family': family_id,
                        'sequence': f"{seq}/{family_data['total_sequences']}",
                        'start': None,
                        'end': None,
                        'lcd': base_time + timedelta(hours=16),
                        'status': 'no_action',
                        'machine': 'Unscheduled'
                    })
        
        # Sort families
        sorted_families = sorted(families.keys())
        
        # Plot jobs
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        for family in sorted_families:
            # Sort sequences
            jobs = sorted(families[family], key=lambda x: int(x['sequence'].split('/')[0]))
            
            # Plot in reverse order (1/3 at bottom)
            for job in reversed(jobs):
                if job['start'] and job['end']:
                    # Scheduled job
                    duration = (job['end'] - job['start']).total_seconds() / 3600
                    start_hour = job['start'].hour + job['start'].minute / 60
                    
                    color = self.colors[job['status']]
                    ax.barh(y_pos, duration, left=start_hour, height=0.8,
                           color=color, edgecolor='black', linewidth=0.5)
                else:
                    # Unscheduled - show as gray bar at end
                    ax.barh(y_pos, 0.5, left=23, height=0.8,
                           color=self.colors['no_action'], edgecolor='black', 
                           linewidth=0.5, alpha=0.5)
                
                # Label
                label = f"{job['family']}_{job['sequence']}"
                y_labels.append(label)
                y_ticks.append(y_pos)
                y_pos += 1
        
        # LCD line
        ax.axvline(x=16, color='red', linestyle='--', linewidth=2, label='LCD')
        
        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Jobs', fontsize=12)
        ax.set_title(f'Production Planning System - {schedule["stage"].upper()} Stage\n' +
                    f'Scheduling Rate: {schedule["scheduling_rate"]:.1%}', 
                    fontsize=16, fontweight='bold')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)
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
        """Create Machine Allocation Gantt chart for a stage."""
        fig, ax = plt.subplots(figsize=(20, 10))
        
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get all machines
        machines = list(schedule['machines'].keys())
        
        y_pos = 0
        y_labels = []
        y_ticks = []
        
        # Plot machine allocations
        for machine in machines:
            jobs = schedule['machines'][machine]
            
            for job in jobs:
                # Calculate times
                start_time = base_time + timedelta(hours=job['start_time'])
                end_time = base_time + timedelta(hours=job['end_time'])
                lcd = base_time + timedelta(hours=job.get('lcd', 16))
                duration = (end_time - start_time).total_seconds() / 3600
                start_hour = start_time.hour + start_time.minute / 60
                
                # Determine status
                status = self.calculate_job_status(end_time, lcd)
                color = self.colors[status]
                
                # Plot bar
                ax.barh(y_pos, duration, left=start_hour, height=0.8,
                       color=color, edgecolor='black', linewidth=0.5)
                
                # Add job text
                job_text = f"{job['family_id']}_{job['sequence_number']}/{job['total_sequences']}"
                bar_center = start_hour + duration / 2
                ax.text(bar_center, y_pos, job_text, ha='center', va='center',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.8))
            
            y_labels.append(machine)
            y_ticks.append(y_pos)
            y_pos += 1
        
        # LCD line
        ax.axvline(x=16, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_title(f'Machine Allocation - {schedule["stage"].upper()} Stage\n' +
                    f'Machines: {len(machines)}, Jobs Scheduled: {schedule["jobs_scheduled"]}',
                    fontsize=16, fontweight='bold')
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=11)
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
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_stage_charts(self):
        """Generate charts for all foundation stages."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for stage in self.foundation_stages:
            print(f"\n{'='*60}")
            print(f"Processing {stage.upper()} stage...")
            print(f"{'='*60}")
            
            # Get schedule from trained model
            schedule = self.get_schedule_from_model(stage)
            
            if schedule:
                # Generate job allocation chart
                job_chart_path = os.path.join(self.output_dir, 
                                            f'q_{stage}_job_allocation_{timestamp}.png')
                self.create_job_allocation_chart(schedule, job_chart_path)
                print(f"✓ Job allocation chart saved: {job_chart_path}")
                
                # Generate machine allocation chart
                machine_chart_path = os.path.join(self.output_dir,
                                                f'q_{stage}_machine_allocation_{timestamp}.png')
                self.create_machine_allocation_chart(schedule, machine_chart_path)
                print(f"✓ Machine allocation chart saved: {machine_chart_path}")
                
                # Print summary
                print(f"\nScheduling Summary:")
                print(f"  - Total jobs: {schedule['total_jobs']}")
                print(f"  - Jobs scheduled: {schedule['jobs_scheduled']}")
                print(f"  - Scheduling rate: {schedule['scheduling_rate']:.1%}")
            else:
                print(f"✗ Could not generate schedule for {stage}")


def main():
    """Main entry point."""
    print("Generating Foundation Stage Schedule Visualizations")
    print("="*60)
    
    visualizer = FoundationScheduleVisualizer()
    visualizer.generate_all_stage_charts()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print(f"Charts saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()