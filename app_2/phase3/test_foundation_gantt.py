"""
Test Foundation Models and Generate Gantt Charts
Shows how each toy stage learned different scheduling patterns
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class FoundationGanttVisualizer:
    """Generate Gantt charts for foundation model schedules."""
    
    def __init__(self):
        self.colors = {
            'on_time': '#4CAF50',      # Green
            'warning': '#FF9800',      # Orange  
            'late': '#F44336',         # Red
            'processing': '#2196F3',   # Blue
            'unscheduled': '#E0E0E0'   # Light gray
        }
        
        self.stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
        
        # Check multiple possible model locations
        self.model_paths = {
            'foundation': "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation",
            'truly_fixed': "/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models",
            'final': "/Users/carrickcheah/Project/ppo/app_2/phase3/final_models"
        }
        
        # Output directory
        self.output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase_3"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.stage_info = {
            'toy_easy': {
                'description': 'Learning sequence dependencies (Job 1 → 2 → 3)',
                'focus': 'Sequential ordering'
            },
            'toy_normal': {
                'description': 'Learning deadlines and priorities',
                'focus': 'Time constraints'
            },
            'toy_hard': {
                'description': 'Complex constraints with late penalties',
                'focus': 'Optimization under pressure'
            },
            'toy_multi': {
                'description': 'Multi-machine job scheduling',
                'focus': 'Resource allocation'
            }
        }
    
    def find_model_path(self, stage: str) -> Optional[str]:
        """Find the model file for a given stage."""
        # Try different locations
        possible_paths = [
            os.path.join(self.model_paths['foundation'], stage, "final_model.zip"),
            os.path.join(self.model_paths['truly_fixed'], f"{stage}_final.zip"),
            os.path.join(self.model_paths['final'], f"{stage}_model.zip")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def test_stage(self, stage: str) -> Dict:
        """Test a stage model and extract schedule."""
        print(f"\nTesting {stage}...")
        
        # Find model
        model_path = self.find_model_path(stage)
        if not model_path:
            print(f"  ✗ No model found for {stage}")
            return None
        
        print(f"  ✓ Found model: {model_path}")
        
        # Load model and create environment
        model = PPO.load(model_path)
        env = CurriculumEnvironmentTrulyFixed(stage, verbose=False)
        
        # Run one episode
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # Track scheduling decisions
        scheduled_jobs = []
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Record successful schedules
            if info.get('action_type') == 'schedule' and info.get('action_valid', False):
                job_info = {
                    'family': info.get('selected_family', 'Unknown'),
                    'sequence': info.get('selected_sequence', 1),
                    'total_seq': info.get('total_sequences', 1),
                    'machine': info.get('selected_machine_name', 'Unknown'),
                    'machine_id': info.get('selected_machine_id', -1),
                    'start': info.get('schedule_start', 0),
                    'end': info.get('schedule_end', 0),
                    'lcd': info.get('lcd', 16),
                    'processing_time': info.get('processing_time', 0)
                }
                scheduled_jobs.append(job_info)
            
            steps += 1
            done = done or truncated
        
        # Calculate metrics
        total_jobs = env.total_tasks
        scheduled = len(env.scheduled_jobs)
        scheduling_rate = scheduled / total_jobs if total_jobs > 0 else 0
        
        # Count on-time vs late
        late_count = 0
        for job in scheduled_jobs:
            if job['end'] > job['lcd']:
                late_count += 1
        
        schedule_data = {
            'stage': stage,
            'jobs': scheduled_jobs,
            'total_jobs': total_jobs,
            'scheduled_count': scheduled,
            'scheduling_rate': scheduling_rate,
            'late_count': late_count,
            'on_time_count': scheduled - late_count,
            'machines': [m['machine_name'] for m in env.machines],
            'families': env.families
        }
        
        print(f"  ✓ Scheduled: {scheduled}/{total_jobs} ({scheduling_rate:.1%})")
        print(f"  ✓ On-time: {scheduled - late_count}, Late: {late_count}")
        
        return schedule_data
    
    def create_gantt_chart(self, schedule: Dict):
        """Create Gantt chart for a stage."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get unique machines
        machines = sorted(list(set(job['machine'] for job in schedule['jobs'])))
        if not machines:
            machines = schedule['machines'][:3]  # Show first 3 machines even if empty
        
        machine_y_pos = {machine: idx for idx, machine in enumerate(machines)}
        
        # Plot scheduled jobs
        for job in schedule['jobs']:
            if job['machine'] in machine_y_pos:
                y_pos = machine_y_pos[job['machine']]
                
                # Determine color
                if job['end'] > job['lcd']:
                    color = self.colors['late']
                elif job['end'] > job['lcd'] - 2:
                    color = self.colors['warning']
                else:
                    color = self.colors['on_time']
                
                # Draw job bar
                duration = job['end'] - job['start']
                ax.barh(y_pos, duration, left=job['start'], height=0.8,
                       color=color, edgecolor='black', linewidth=1)
                
                # Add job label
                label = f"{job['family']}_S{job['sequence']}"
                if duration > 0.5:  # Only if wide enough
                    ax.text(job['start'] + duration/2, y_pos, label,
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # LCD line
        ax.axvline(x=16, color='red', linestyle='--', linewidth=2, label='LCD (16h)')
        
        # Formatting
        stage_name = schedule['stage']
        stage_desc = self.stage_info[stage_name]['description']
        stage_focus = self.stage_info[stage_name]['focus']
        
        ax.set_title(f'Foundation Model: {stage_name.upper()} Stage\n'
                    f'{stage_desc}\n'
                    f'Focus: {stage_focus}',
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_xlim(0, 24)
        ax.set_ylim(-0.5, len(machines) - 0.5)
        
        # Y-axis
        ax.set_yticks(list(range(len(machines))))
        ax.set_yticklabels(machines)
        
        # X-axis
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)], rotation=45)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors['on_time'], label='On-time'),
            mpatches.Patch(color=self.colors['warning'], label='Warning (<2h)'),
            mpatches.Patch(color=self.colors['late'], label='Late'),
            mpatches.Line2D([0], [0], color='red', linestyle='--', label='LCD')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Performance box
        perf_text = (f"Scheduling Rate: {schedule['scheduling_rate']:.1%}\n"
                    f"Jobs: {schedule['scheduled_count']}/{schedule['total_jobs']}\n"
                    f"On-time: {schedule['on_time_count']} | Late: {schedule['late_count']}")
        ax.text(0.02, 0.98, perf_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'foundation_{stage_name}_gantt_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
        
        return filepath
    
    def create_comparison_chart(self, results: List[Dict]):
        """Create comparison chart of all stages."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        stages = [r['stage'] for r in results if r]
        rates = [r['scheduling_rate'] * 100 for r in results if r]
        on_time_rates = [(r['on_time_count'] / r['scheduled_count'] * 100 
                         if r['scheduled_count'] > 0 else 0) for r in results if r]
        
        # Color mapping
        colors = {
            'toy_easy': '#4CAF50',
            'toy_normal': '#2196F3', 
            'toy_hard': '#FF9800',
            'toy_multi': '#9C27B0'
        }
        
        bar_colors = [colors.get(s, '#666666') for s in stages]
        
        # 1. Scheduling Rate
        bars1 = ax1.bar(stages, rates, color=bar_colors)
        ax1.set_ylabel('Scheduling Rate (%)')
        ax1.set_title('Scheduling Success Rate')
        ax1.set_ylim(0, 105)
        for bar, rate in zip(bars1, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 2. On-time Rate
        bars2 = ax2.bar(stages, on_time_rates, color=bar_colors)
        ax2.set_ylabel('On-time Rate (%)')
        ax2.set_title('On-time Completion Rate')
        ax2.set_ylim(0, 105)
        for bar, rate in zip(bars2, on_time_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Jobs Scheduled
        scheduled = [r['scheduled_count'] for r in results if r]
        total = [r['total_jobs'] for r in results if r]
        x = np.arange(len(stages))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, scheduled, width, label='Scheduled', color='#2196F3')
        bars3b = ax3.bar(x + width/2, total, width, label='Total', color='#E0E0E0')
        ax3.set_ylabel('Number of Jobs')
        ax3.set_title('Jobs Scheduled vs Total')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stages)
        ax3.legend()
        
        # 4. Stage Focus Summary
        ax4.axis('off')
        summary_text = "Stage Learning Focus:\n\n"
        for stage in self.stages:
            if stage in stages:
                info = self.stage_info[stage]
                summary_text += f"• {stage.upper()}:\n"
                summary_text += f"  {info['focus']}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        
        plt.suptitle('Foundation Model Training Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'foundation_comparison_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Comparison chart saved: {filename}")
    
    def visualize_all(self):
        """Test all stages and create visualizations."""
        print("="*60)
        print("Foundation Model Gantt Chart Generation")
        print("="*60)
        
        results = []
        
        for stage in self.stages:
            schedule = self.test_stage(stage)
            if schedule:
                self.create_gantt_chart(schedule)
                results.append(schedule)
        
        # Create comparison if we have results
        if results:
            self.create_comparison_chart(results)
        
        print("\n" + "="*60)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print("="*60)


def main():
    visualizer = FoundationGanttVisualizer()
    visualizer.visualize_all()


if __name__ == "__main__":
    main()