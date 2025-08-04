"""
Comprehensive Phase 4 PPO Model Testing with Enhanced Gantt Visualization
Tests all available trained models and creates detailed Gantt charts
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase4.environments import (
    SmallBalancedEnvironment,
    SmallRushEnvironment,
    SmallBottleneckEnvironment,
    SmallComplexEnvironment
)


class EnhancedPhase4Visualizer:
    """Enhanced Gantt chart creator for Phase 4 with detailed metrics."""
    
    def __init__(self):
        self.colors = {
            'late': '#FF4444',      # Red for late jobs
            'warning': '#FF8800',   # Orange for jobs at risk
            'caution': '#FFD700',   # Gold for jobs with tight deadlines
            'ok': '#44AA44'         # Green for jobs on time
        }
        
        # Machine colors for variety
        self.machine_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
    
    def _get_deadline_status(self, end_time: float, lcd_date: str, current_date: str = "2025-07-25") -> str:
        """Determine deadline status for color coding."""
        try:
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            job_end_dt = current_dt + timedelta(hours=end_time)
            lcd_dt = datetime.strptime(lcd_date, "%Y-%m-%d")
            
            days_diff = (lcd_dt - job_end_dt).days
            
            if days_diff < 0:
                return 'late'
            elif days_diff <= 1:
                return 'warning'
            elif days_diff <= 3:
                return 'caution'
            else:
                return 'ok'
        except:
            return 'ok'
    
    def create_enhanced_job_allocation_chart(self, schedule_data: Dict, save_path: str, scenario: str, model_info: Dict):
        """Create enhanced job allocation Gantt chart with detailed metrics."""
        
        if not schedule_data or 'families' not in schedule_data:
            print(f"No valid schedule data for {scenario}")
            return
        
        # Extract scheduled jobs
        scheduled_jobs = []
        total_jobs = 0
        
        for family_id, family_data in schedule_data['families'].items():
            total_jobs += 1
            if 'scheduled_tasks' in family_data and family_data['scheduled_tasks']:
                for task in family_data['scheduled_tasks']:
                    if 'start_time' in task and 'end_time' in task:
                        scheduled_jobs.append({
                            'job_id': family_id,
                            'task_id': f"{family_id}_seq{task.get('sequence', 1)}",
                            'start': task['start_time'],
                            'end': task['end_time'],
                            'machine': task.get('machine_id', 'Unknown'),
                            'lcd_date': family_data.get('lcd_date', '2025-08-15'),
                            'process': task.get('process_name', 'Unknown')
                        })
        
        if not scheduled_jobs:
            # Create empty chart with message
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.text(0.5, 0.5, f'No jobs scheduled for {scenario}\\nModel: {model_info.get("checkpoint", "Unknown")}', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title(f'Phase 4 Job Allocation - {scenario.replace("_", " ").title()}\\nNo Scheduling Results', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Empty job allocation chart saved: {save_path}")
            return
        
        # Sort jobs by job_id and start time
        scheduled_jobs.sort(key=lambda x: (x['job_id'], x['start']))
        
        # Create figure with metrics subplot
        fig = plt.figure(figsize=(18, max(10, len(scheduled_jobs) * 0.4)))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
        
        # Main Gantt chart
        ax_main = fig.add_subplot(gs[0, :])
        
        # Plot job bars
        y_positions = {}
        current_y = 0
        
        # Count jobs by status
        status_counts = {'late': 0, 'warning': 0, 'caution': 0, 'ok': 0}
        
        for job in scheduled_jobs:
            job_id = job['job_id']
            if job_id not in y_positions:
                y_positions[job_id] = current_y
                current_y += 1
            
            y_pos = y_positions[job_id]
            duration = job['end'] - job['start']
            
            # Get color based on deadline status
            status = self._get_deadline_status(job['end'], job['lcd_date'])
            color = self.colors[status]
            status_counts[status] += 1
            
            # Create bar
            bar = ax_main.barh(y_pos, duration, left=job['start'], height=0.6, 
                              color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add task label
            if duration > 5:  # Only show label if bar is wide enough
                ax_main.text(job['start'] + duration/2, y_pos, f"M{job['machine']}", 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Current time marker
        if scheduled_jobs:
            max_time = max([job['end'] for job in scheduled_jobs])
            current_time = max_time * 0.1
            ax_main.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Time')
        
        # Formatting main chart
        ax_main.set_yticks(list(y_positions.values()))
        ax_main.set_yticklabels([jid[:15] + '...' if len(jid) > 15 else jid for jid in y_positions.keys()], fontsize=9)
        ax_main.set_xlabel('Time (hours)', fontsize=12)
        ax_main.set_ylabel('Job IDs', fontsize=12)
        
        # Enhanced title with metrics
        completion_rate = len(scheduled_jobs) / total_jobs if total_jobs > 0 else 0
        title = f'Phase 4 Job Allocation - {scenario.replace("_", " ").title()}\\n'
        title += f'Completion Rate: {completion_rate:.1%} ({len(scheduled_jobs)}/{total_jobs}) | '
        title += f'Model: {model_info.get("checkpoint", "Unknown")} | Reward: {schedule_data.get("total_reward", 0):.0f}'
        ax_main.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            patches.Patch(color=self.colors['late'], label=f'Late Jobs ({status_counts["late"]})'),
            patches.Patch(color=self.colors['warning'], label=f'Warning ≤1d ({status_counts["warning"]})'),
            patches.Patch(color=self.colors['caution'], label=f'Caution ≤3d ({status_counts["caution"]})'),
            patches.Patch(color=self.colors['ok'], label=f'On Time ({status_counts["ok"]})')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_axisbelow(True)
        
        # Metrics subplots
        ax_metrics1 = fig.add_subplot(gs[1, 0])
        ax_metrics2 = fig.add_subplot(gs[1, 1])
        
        # Status distribution pie chart
        if sum(status_counts.values()) > 0:
            labels = [f'{k.title()}\\n({v})' for k, v in status_counts.items() if v > 0]
            sizes = [v for v in status_counts.values() if v > 0]
            colors = [self.colors[k] for k in status_counts.keys() if status_counts[k] > 0]
            
            ax_metrics1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax_metrics1.set_title('Deadline Status Distribution', fontsize=12, fontweight='bold')
        
        # Model performance metrics
        metrics_text = f"""Model Performance:
• Checkpoint: {model_info.get('checkpoint', 'Unknown')}
• Steps Trained: {model_info.get('steps', 'N/A')}
• Total Reward: {schedule_data.get('total_reward', 0):.0f}
• Families Scheduled: {len(scheduled_jobs)} / {total_jobs}
• Completion Rate: {completion_rate:.1%}
• Average Duration: {np.mean([j['end'] - j['start'] for j in scheduled_jobs]):.1f}h"""
        
        ax_metrics2.text(0.05, 0.95, metrics_text, transform=ax_metrics2.transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax_metrics2.set_xlim(0, 1)
        ax_metrics2.set_ylim(0, 1)
        ax_metrics2.axis('off')
        ax_metrics2.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced job allocation chart saved: {save_path}")
    
    def create_enhanced_machine_allocation_chart(self, schedule_data: Dict, save_path: str, scenario: str, model_info: Dict):
        """Create enhanced machine allocation Gantt chart with utilization metrics."""
        
        if not schedule_data or 'families' not in schedule_data:
            print(f"No valid schedule data for {scenario}")
            return
        
        # Extract machine schedules
        machine_schedules = {}
        machine_names = {}
        
        for family_id, family_data in schedule_data['families'].items():
            if 'scheduled_tasks' in family_data and family_data['scheduled_tasks']:
                for task in family_data['scheduled_tasks']:
                    if 'start_time' in task and 'end_time' in task:
                        machine_id = task.get('machine_id', 'Unknown')
                        if machine_id not in machine_schedules:
                            machine_schedules[machine_id] = []
                            machine_names[machine_id] = f"Machine {machine_id}"
                        
                        machine_schedules[machine_id].append({
                            'job_id': family_id,
                            'start': task['start_time'],
                            'end': task['end_time'],
                            'lcd_date': family_data.get('lcd_date', '2025-08-15'),
                            'process': task.get('process_name', 'Unknown'),
                            'sequence': task.get('sequence', 1)
                        })
        
        if not machine_schedules:
            # Create empty chart
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.text(0.5, 0.5, f'No machine schedules for {scenario}\\nModel: {model_info.get("checkpoint", "Unknown")}', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title(f'Phase 4 Machine Allocation - {scenario.replace("_", " ").title()}\\nNo Scheduling Results', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Empty machine allocation chart saved: {save_path}")
            return
        
        # Sort machines by ID
        sorted_machines = sorted(machine_schedules.keys())
        
        # Create figure with utilization metrics
        fig = plt.figure(figsize=(18, max(10, len(sorted_machines) * 0.8)))
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1], wspace=0.3)
        
        # Main Gantt chart
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Calculate overall metrics
        total_busy_time = 0
        max_time = 0
        utilizations = []
        
        # Plot machine schedules
        for i, machine_id in enumerate(sorted_machines):
            jobs = machine_schedules[machine_id]
            jobs.sort(key=lambda x: x['start'])  # Sort by start time
            
            # Calculate utilization
            machine_busy_time = sum(job['end'] - job['start'] for job in jobs)
            machine_max_time = max([job['end'] for job in jobs]) if jobs else 0
            utilization = (machine_busy_time / machine_max_time * 100) if machine_max_time > 0 else 0
            utilizations.append(utilization)
            
            total_busy_time += machine_busy_time
            max_time = max(max_time, machine_max_time)
            
            for job in jobs:
                duration = job['end'] - job['start']
                
                # Get color based on deadline status
                status = self._get_deadline_status(job['end'], job['lcd_date'])
                color = self.colors[status]
                
                # Create bar
                bar = ax_main.barh(i, duration, left=job['start'], height=0.6,
                                 color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add job label
                if duration > 3:  # Only show label if bar is wide enough
                    label = f"{job['job_id'][:8]}..."  # Truncate long job IDs
                    ax_main.text(job['start'] + duration/2, i, label,
                               ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add utilization percentage
            ax_main.text(-max_time*0.05, i, f"{utilization:.1f}%", ha='right', va='center', 
                       fontsize=10, fontweight='bold')
        
        # Formatting main chart
        ax_main.set_yticks(range(len(sorted_machines)))
        ax_main.set_yticklabels([machine_names[m] for m in sorted_machines], fontsize=10)
        ax_main.set_xlabel('Time (hours)', fontsize=12)
        ax_main.set_ylabel('Machines (Utilization %)', fontsize=12)
        
        # Enhanced title
        overall_utilization = (total_busy_time / (max_time * len(sorted_machines)) * 100) if max_time > 0 else 0
        title = f'Phase 4 Machine Allocation - {scenario.replace("_", " ").title()}\\n'
        title += f'Overall Utilization: {overall_utilization:.1f}% | Active Machines: {len(sorted_machines)} | '
        title += f'Model: {model_info.get("checkpoint", "Unknown")}'
        ax_main.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            patches.Patch(color=self.colors['late'], label='Late Jobs'),
            patches.Patch(color=self.colors['warning'], label='Warning (≤1 day)'),
            patches.Patch(color=self.colors['caution'], label='Caution (≤3 days)'),
            patches.Patch(color=self.colors['ok'], label='On Time')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_axisbelow(True)
        
        # Utilization histogram
        ax_util = fig.add_subplot(gs[0, 1])
        if utilizations:
            ax_util.hist(utilizations, bins=min(10, len(utilizations)), alpha=0.7, color='skyblue', edgecolor='black')
            ax_util.set_xlabel('Utilization (%)', fontsize=10)
            ax_util.set_ylabel('Machine Count', fontsize=10)
            ax_util.set_title('Utilization Distribution', fontsize=12, fontweight='bold')
            ax_util.grid(True, alpha=0.3)
        
        # Metrics summary
        ax_metrics = fig.add_subplot(gs[0, 2])
        metrics_text = f"""Machine Metrics:
• Active Machines: {len(sorted_machines)}
• Avg Utilization: {np.mean(utilizations):.1f}%
• Max Utilization: {max(utilizations):.1f}%
• Min Utilization: {min(utilizations):.1f}%
• Overall Schedule: {overall_utilization:.1f}%

Model Info:
• Checkpoint: {model_info.get('checkpoint', 'Unknown')}
• Total Reward: {schedule_data.get('total_reward', 0):.0f}
• Max Time: {max_time:.1f}h"""
        
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)
        ax_metrics.axis('off')
        ax_metrics.set_title('Performance Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced machine allocation chart saved: {save_path}")


def run_ppo_model_comprehensive(model_path: str, env_class, max_steps: int = 200) -> Tuple[Dict, Dict]:
    """Run trained PPO model and return both schedule data and model info."""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return {}, {}
    
    # Extract model info from path
    model_filename = os.path.basename(model_path)
    if 'checkpoint' in model_filename:
        steps = model_filename.split('_')[-2] if '_' in model_filename else 'unknown'
        checkpoint = f"checkpoint_{steps}_steps"
    elif 'final' in model_filename:
        checkpoint = "final_model"
    else:
        checkpoint = model_filename.replace('.zip', '')
    
    model_info = {
        'path': model_path,
        'checkpoint': checkpoint,
        'steps': steps if 'checkpoint' in model_filename else 'final'
    }
    
    try:
        # Load trained PPO model
        print(f"Loading PPO model: {checkpoint}")
        model = PPO.load(model_path)
        
        # Create environment
        env = env_class(verbose=False)
        
        # Generate schedule using trained model
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Use trained model to predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        # Extract schedule data
        schedule_data = {
            'scenario': getattr(env, 'scenario_name', 'unknown'),
            'total_reward': total_reward,
            'steps_taken': step + 1,
            'families': {}
        }
        
        # Extract family and task data
        for family_id, family_data in env.families.items():
            family_info = {
                'job_reference': family_data['job_reference'],
                'lcd_date': family_data['lcd_date'],
                'scheduled_tasks': []
            }
            schedule_data['families'][family_id] = family_info
        
        # Get scheduled tasks from machine schedules
        for machine_id, machine_jobs in env.machine_schedules.items():
            for job in machine_jobs:
                job_key = job['job']  # Format: 'family_id_seqN'
                if '_seq' in job_key:
                    family_id = job_key.split('_seq')[0]
                    sequence = int(job_key.split('_seq')[1])
                    
                    if family_id in schedule_data['families']:
                        schedule_data['families'][family_id]['scheduled_tasks'].append({
                            'sequence': sequence,
                            'process_name': f"Process_seq{sequence}",
                            'start_time': job['start'],
                            'end_time': job['end'],
                            'machine_id': machine_id
                        })
        
        print(f"Schedule generated - Reward: {total_reward:.1f}, Steps: {step + 1}")
        return schedule_data, model_info
        
    except Exception as e:
        print(f"Error running PPO model: {e}")
        import traceback
        traceback.print_exc()
        return {}, model_info


def main():
    """Comprehensive Phase 4 testing with all available models."""
    
    print("="*80)
    print("COMPREHENSIVE PHASE 4 PPO MODEL TESTING")
    print("="*80)
    
    # Initialize enhanced visualizer
    visualizer = EnhancedPhase4Visualizer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = '/Users/carrickcheah/Project/ppo/app_2/visualizations/phase4'
    
    # Define all available scenarios with their best models
    scenarios = [
        {
            'name': 'small_balanced',
            'env_class': SmallBalancedEnvironment,
            'models': [
                '/Users/carrickcheah/Project/ppo/app_2/phase4/results/small_balanced/small_balanced_final.zip',
                '/Users/carrickcheah/Project/ppo/app_2/phase4/results/small_balanced/checkpoints/small_balanced_checkpoint_500000_steps.zip'
            ]
        },
        {
            'name': 'small_rush',
            'env_class': SmallRushEnvironment,
            'models': [
                '/Users/carrickcheah/Project/ppo/app_2/phase4/results/small_rush/checkpoints/small_rush_checkpoint_300000_steps.zip',
                '/Users/carrickcheah/Project/ppo/app_2/phase4/results/small_rush/checkpoints/small_rush_checkpoint_250000_steps.zip'
            ]
        }
    ]
    
    results_summary = {
        'timestamp': timestamp,
        'comprehensive_test': True,
        'scenarios_tested': [],
        'total_models_tested': 0,
        'visualizations_created': []
    }
    
    for scenario in scenarios:
        print(f"\\n{'='*60}")
        print(f"Testing scenario: {scenario['name'].upper()}")
        print(f"{'='*60}")
        
        scenario_results = {
            'scenario': scenario['name'],
            'models_tested': [],
            'best_reward': float('-inf'),
            'best_model': None
        }
        
        for model_path in scenario['models']:
            if not os.path.exists(model_path):
                print(f"Skipping missing model: {model_path}")
                continue
            
            print(f"\\nTesting model: {os.path.basename(model_path)}")
            print("-" * 40)
            
            # Run PPO model to generate schedule
            schedule_data, model_info = run_ppo_model_comprehensive(
                model_path, scenario['env_class']
            )
            
            if not schedule_data:
                print(f"Failed to generate schedule")
                continue
            
            # Track best model
            reward = schedule_data.get('total_reward', float('-inf'))
            if reward > scenario_results['best_reward']:
                scenario_results['best_reward'] = reward
                scenario_results['best_model'] = model_info['checkpoint']
            
            # Create enhanced visualizations
            base_name = f"comprehensive_{scenario['name']}_{model_info['checkpoint']}_{timestamp}"
            
            # Job allocation chart
            job_chart_path = os.path.join(viz_dir, f"{base_name}_job_allocation.png")
            visualizer.create_enhanced_job_allocation_chart(
                schedule_data, job_chart_path, scenario['name'], model_info
            )
            
            # Machine allocation chart
            machine_chart_path = os.path.join(viz_dir, f"{base_name}_machine_allocation.png")
            visualizer.create_enhanced_machine_allocation_chart(
                schedule_data, machine_chart_path, scenario['name'], model_info
            )
            
            # Save detailed schedule data
            schedule_path = os.path.join(viz_dir, f"{base_name}_schedule.json")
            detailed_data = {
                'model_info': model_info,
                'schedule_data': schedule_data,
                'test_timestamp': timestamp
            }
            with open(schedule_path, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            
            # Update results
            model_result = {
                'model_path': model_path,
                'checkpoint': model_info['checkpoint'],
                'total_reward': reward,
                'steps_taken': schedule_data.get('steps_taken', 0),
                'scheduled_families': len([f for f in schedule_data.get('families', {}).values() 
                                         if f.get('scheduled_tasks', [])])
            }
            scenario_results['models_tested'].append(model_result)
            results_summary['total_models_tested'] += 1
            
            results_summary['visualizations_created'].extend([
                job_chart_path, machine_chart_path, schedule_path
            ])
            
            print(f"✓ Completed {model_info['checkpoint']} - Reward: {reward:.1f}")
        
        results_summary['scenarios_tested'].append(scenario_results)
        print(f"\\n✓ Scenario {scenario['name']} completed - Best reward: {scenario_results['best_reward']:.1f}")
    
    # Save comprehensive summary
    summary_path = os.path.join(viz_dir, f"comprehensive_phase4_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\\n" + "="*80)
    print("COMPREHENSIVE PHASE 4 TESTING COMPLETED")
    print("="*80)
    print(f"Total models tested: {results_summary['total_models_tested']}")
    print(f"Total scenarios: {len(results_summary['scenarios_tested'])}")
    print(f"Total visualizations: {len(results_summary['visualizations_created'])}")
    print(f"Summary saved: {summary_path}")
    
    # Print best performers
    print("\\nBest Performers by Scenario:")
    for scenario_result in results_summary['scenarios_tested']:
        print(f"  {scenario_result['scenario']}: {scenario_result['best_model']} "
              f"(Reward: {scenario_result['best_reward']:.1f})")


if __name__ == "__main__":
    main()