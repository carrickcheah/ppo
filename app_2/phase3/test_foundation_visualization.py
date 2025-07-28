"""
Test Foundation Stage Models and Generate Gantt Charts
Visualizes how each toy stage model learned different scheduling rules
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
from phase3.environments.curriculum_env_fixed import CurriculumEnvironmentFixed


class FoundationModelTester:
    """Test foundation models and generate Gantt chart visualizations."""
    
    def __init__(self):
        """Initialize tester with correct paths."""
        self.colors = {
            'on_time': '#4CAF50',      # Green - completed well before deadline
            'warning': '#FF9800',      # Orange - close to deadline  
            'late': '#F44336',         # Red - past deadline
            'processing': '#2196F3',   # Blue - currently processing
            'unscheduled': '#E0E0E0'   # Light gray - not scheduled
        }
        
        self.foundation_stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
        self.checkpoint_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation"
        
        # Create proper output directory
        self.output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase_3"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store stage descriptions
        self.stage_descriptions = {
            'toy_easy': 'Learning basic sequence rules (1→2→3)',
            'toy_normal': 'Learning deadlines and priorities',
            'toy_hard': 'Learning complex constraints and late penalties',
            'toy_multi': 'Learning multi-machine job scheduling'
        }
    
    def test_model_and_get_schedule(self, stage_name: str, num_episodes: int = 3) -> Dict:
        """Test a trained model and extract scheduling results."""
        print(f"\nTesting {stage_name} model...")
        
        # Create environment
        base_env = CurriculumEnvironmentFixed(stage_name=stage_name, verbose=False)
        env = Monitor(base_env)
        
        # Load trained model
        model_path = os.path.join(self.checkpoint_dir, stage_name, "final_model.zip")
        if not os.path.exists(model_path):
            print(f"Warning: No model found at {model_path}")
            return None
        
        model = PPO.load(model_path)
        
        # Run multiple episodes to get average performance
        all_schedules = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            scheduled_jobs = []
            machine_schedules = {m['machine_name']: [] for m in base_env.machines}
            
            step = 0
            while not done and step < 200:  # Limit steps
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                # Record scheduled job if action was valid
                if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                    job_info = {
                        'family_id': info.get('selected_family', ''),
                        'sequence_number': info.get('selected_sequence', 1),
                        'total_sequences': info.get('total_sequences', 1),
                        'machine_id': info.get('selected_machine_id', -1),
                        'machine_name': info.get('selected_machine_name', 'Unknown'),
                        'start_time': info.get('schedule_start', 0),
                        'end_time': info.get('schedule_end', 0),
                        'processing_time': info.get('processing_time', 0),
                        'lcd': info.get('lcd', 16),
                        'required_machines': info.get('required_machines', [])
                    }
                    scheduled_jobs.append(job_info)
                    
                    # Add to machine schedules
                    if job_info['machine_name'] in machine_schedules:
                        machine_schedules[job_info['machine_name']].append(job_info)
                
                step += 1
                done = done or truncated
            
            # Calculate metrics
            total_jobs = base_env.total_tasks
            jobs_scheduled = len(base_env.scheduled_jobs)
            scheduling_rate = jobs_scheduled / total_jobs if total_jobs > 0 else 0
            
            # Check for late jobs
            late_jobs = 0
            on_time_jobs = 0
            for job in scheduled_jobs:
                if job['end_time'] > job['lcd']:
                    late_jobs += 1
                else:
                    on_time_jobs += 1
            
            episode_schedule = {
                'jobs': scheduled_jobs,
                'machines': machine_schedules,
                'total_jobs': total_jobs,
                'jobs_scheduled': jobs_scheduled,
                'scheduling_rate': scheduling_rate,
                'late_jobs': late_jobs,
                'on_time_jobs': on_time_jobs,
                'total_reward': env.get_wrapper_attr('episode_returns', default=[0])[-1] if hasattr(env, 'get_wrapper_attr') else 0
            }
            all_schedules.append(episode_schedule)
        
        # Return best episode (highest scheduling rate)
        best_schedule = max(all_schedules, key=lambda x: x['scheduling_rate'])
        best_schedule['stage'] = stage_name
        best_schedule['families'] = base_env.families
        
        return best_schedule
    
    def create_gantt_chart(self, schedule: Dict, output_path: str):
        """Create a Gantt chart showing the scheduling results."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get all machines and sort them
        machines = sorted(schedule['machines'].keys())
        machine_y_pos = {machine: idx for idx, machine in enumerate(machines)}
        
        # Base time for visualization
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Plot jobs on each machine
        for machine, jobs in schedule['machines'].items():
            y_pos = machine_y_pos[machine]
            
            for job in jobs:
                # Calculate display times
                start_hour = job['start_time']
                duration = job['end_time'] - job['start_time']
                
                # Determine color based on deadline
                if job['end_time'] > job['lcd']:
                    color = self.colors['late']
                elif job['end_time'] > job['lcd'] - 2:  # Within 2 hours of deadline
                    color = self.colors['warning']
                else:
                    color = self.colors['on_time']
                
                # Draw job bar
                rect = ax.barh(y_pos, duration, left=start_hour, height=0.8,
                             color=color, edgecolor='black', linewidth=1)
                
                # Add job label
                job_label = f"{job['family_id']}_S{job['sequence_number']}"
                if duration > 1:  # Only add text if bar is wide enough
                    ax.text(start_hour + duration/2, y_pos, job_label, 
                           ha='center', va='center', fontsize=8, fontweight='bold')
                
                # Draw LCD line for this job
                if job['lcd'] < 24:  # Only show if within chart range
                    ax.plot([job['lcd'], job['lcd']], [y_pos - 0.4, y_pos + 0.4], 
                           'r--', linewidth=1, alpha=0.7)
        
        # Add unscheduled jobs indicator
        unscheduled_count = schedule['total_jobs'] - schedule['jobs_scheduled']
        if unscheduled_count > 0:
            ax.text(0.02, 0.98, f"Unscheduled: {unscheduled_count} jobs", 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5),
                   verticalalignment='top')
        
        # Main LCD line at hour 16
        ax.axvline(x=16, color='red', linestyle='--', linewidth=2, alpha=0.7, label='LCD (16h)')
        
        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.5, len(machines) - 0.5)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        
        # Title with stage description
        stage_desc = self.stage_descriptions.get(schedule['stage'], '')
        ax.set_title(f'Foundation Model Schedule - {schedule["stage"].upper()} Stage\n'
                    f'{stage_desc}\n'
                    f'Scheduling Rate: {schedule["scheduling_rate"]:.1%} | '
                    f'On-time: {schedule["on_time_jobs"]} | Late: {schedule["late_jobs"]}',
                    fontsize=14, fontweight='bold')
        
        # Set y-axis labels
        ax.set_yticks(list(range(len(machines))))
        ax.set_yticklabels(machines, fontsize=10)
        
        # Set x-axis
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)], rotation=45)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['on_time'], label='On-time (>2h buffer)'),
            mpatches.Patch(color=self.colors['warning'], label='Warning (<2h buffer)'),
            mpatches.Patch(color=self.colors['late'], label='Late (past LCD)'),
            mpatches.Patch(color='none', edgecolor='red', linestyle='--', label='LCD deadline')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add performance metrics box
        metrics_text = (f"Total Jobs: {schedule['total_jobs']}\n"
                       f"Scheduled: {schedule['jobs_scheduled']}\n"
                       f"Scheduling Rate: {schedule['scheduling_rate']:.1%}")
        ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gantt chart saved: {output_path}")
    
    def test_all_stages(self):
        """Test all foundation stage models and generate visualizations."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("Testing Foundation Stage Models")
        print("=" * 60)
        
        summary_results = []
        
        for stage in self.foundation_stages:
            print(f"\n{'='*60}")
            print(f"Testing {stage.upper()} stage")
            print(f"Description: {self.stage_descriptions[stage]}")
            print(f"{'='*60}")
            
            # Test model and get schedule
            schedule = self.test_model_and_get_schedule(stage, num_episodes=3)
            
            if schedule:
                # Generate Gantt chart
                gantt_path = os.path.join(self.output_dir, 
                                         f'foundation_{stage}_gantt_{timestamp}.png')
                self.create_gantt_chart(schedule, gantt_path)
                
                # Print performance summary
                print(f"\nPerformance Summary for {stage}:")
                print(f"  - Total jobs: {schedule['total_jobs']}")
                print(f"  - Jobs scheduled: {schedule['jobs_scheduled']}")
                print(f"  - Scheduling rate: {schedule['scheduling_rate']:.1%}")
                print(f"  - On-time jobs: {schedule['on_time_jobs']}")
                print(f"  - Late jobs: {schedule['late_jobs']}")
                
                summary_results.append({
                    'stage': stage,
                    'scheduling_rate': schedule['scheduling_rate'],
                    'on_time_rate': schedule['on_time_jobs'] / schedule['jobs_scheduled'] if schedule['jobs_scheduled'] > 0 else 0,
                    'total_jobs': schedule['total_jobs'],
                    'scheduled_jobs': schedule['jobs_scheduled']
                })
            else:
                print(f"✗ Could not test {stage} model")
        
        # Create comparison chart
        if summary_results:
            self.create_comparison_chart(summary_results, timestamp)
        
        return summary_results
    
    def create_comparison_chart(self, results: List[Dict], timestamp: str):
        """Create a comparison chart of all stages."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stages = [r['stage'] for r in results]
        scheduling_rates = [r['scheduling_rate'] * 100 for r in results]
        on_time_rates = [r['on_time_rate'] * 100 for r in results]
        
        # Scheduling rate comparison
        bars1 = ax1.bar(stages, scheduling_rates, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
        ax1.set_xlabel('Stage', fontsize=12)
        ax1.set_ylabel('Scheduling Rate (%)', fontsize=12)
        ax1.set_title('Scheduling Success Rate by Stage', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # On-time rate comparison
        bars2 = ax2.bar(stages, on_time_rates, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
        ax2.set_xlabel('Stage', fontsize=12)
        ax2.set_ylabel('On-time Rate (%)', fontsize=12)
        ax2.set_title('On-time Completion Rate by Stage', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Foundation Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = os.path.join(self.output_dir, f'foundation_comparison_{timestamp}.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison chart saved: {comparison_path}")


def main():
    """Main entry point."""
    tester = FoundationModelTester()
    results = tester.test_all_stages()
    
    print("\n" + "="*60)
    print("Foundation Model Testing Complete!")
    print(f"All visualizations saved to: {tester.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()