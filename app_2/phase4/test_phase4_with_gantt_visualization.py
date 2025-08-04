"""
Phase 4 PPO Model Testing with Gantt Chart Visualization
Uses trained PPO models to schedule jobs and creates comprehensive Gantt charts
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


class Phase4ScheduleVisualizer:
    """Create professional Gantt charts for Phase 4 scheduling results."""
    
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
    
    def create_job_allocation_chart(self, schedule_data: Dict, save_path: str, scenario: str):
        """Create job allocation Gantt chart (job-view)."""
        
        if not schedule_data or 'families' not in schedule_data:
            print(f"No valid schedule data for {scenario}")
            return
        
        # Extract scheduled jobs
        scheduled_jobs = []
        for family_id, family_data in schedule_data['families'].items():
            if 'scheduled_tasks' in family_data:
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
            print(f"No scheduled jobs found for {scenario}")
            return
        
        # Sort jobs by job_id
        scheduled_jobs.sort(key=lambda x: x['job_id'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(8, len(scheduled_jobs) * 0.4)))
        
        # Plot job bars
        y_positions = {}
        current_y = 0
        
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
            
            # Create bar
            bar = ax.barh(y_pos, duration, left=job['start'], height=0.6, 
                         color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add task label
            if duration > 5:  # Only show label if bar is wide enough
                ax.text(job['start'] + duration/2, y_pos, f"M{job['machine']}", 
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Current time marker
        current_time = max([job['end'] for job in scheduled_jobs]) * 0.1  # Assume we're 10% through
        ax.axvline(x=current_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Current Time')
        
        # Formatting
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()), fontsize=10)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Job IDs', fontsize=12)
        ax.set_title(f'Phase 4 Job Allocation - {scenario.replace("_", " ").title()}\nJob Timeline View', 
                    fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            patches.Patch(color=self.colors['late'], label='Late Jobs'),
            patches.Patch(color=self.colors['warning'], label='Warning (≤1 day)'),
            patches.Patch(color=self.colors['caution'], label='Caution (≤3 days)'),
            patches.Patch(color=self.colors['ok'], label='On Time')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Job allocation chart saved: {save_path}")
    
    def create_machine_allocation_chart(self, schedule_data: Dict, save_path: str, scenario: str):
        """Create machine allocation Gantt chart (machine-view)."""
        
        if not schedule_data or 'families' not in schedule_data:
            print(f"No valid schedule data for {scenario}")
            return
        
        # Extract machine schedules
        machine_schedules = {}
        machine_names = {}
        
        for family_id, family_data in schedule_data['families'].items():
            if 'scheduled_tasks' in family_data:
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
            print(f"No machine schedules found for {scenario}")
            return
        
        # Sort machines by ID
        sorted_machines = sorted(machine_schedules.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(8, len(sorted_machines) * 0.6)))
        
        # Plot machine schedules
        for i, machine_id in enumerate(sorted_machines):
            jobs = machine_schedules[machine_id]
            jobs.sort(key=lambda x: x['start'])  # Sort by start time
            
            # Calculate utilization
            total_busy_time = sum(job['end'] - job['start'] for job in jobs)
            max_time = max([job['end'] for job in jobs]) if jobs else 0
            utilization = (total_busy_time / max_time * 100) if max_time > 0 else 0
            
            for job in jobs:
                duration = job['end'] - job['start']
                
                # Get color based on deadline status
                status = self._get_deadline_status(job['end'], job['lcd_date'])
                color = self.colors[status]
                
                # Create bar
                bar = ax.barh(i, duration, left=job['start'], height=0.6,
                             color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Add job label
                if duration > 3:  # Only show label if bar is wide enough
                    label = f"{job['job_id'][:8]}..."  # Truncate long job IDs
                    ax.text(job['start'] + duration/2, i, label,
                           ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add utilization percentage
            ax.text(-5, i, f"{utilization:.1f}%", ha='right', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_yticks(range(len(sorted_machines)))
        ax.set_yticklabels([machine_names[m] for m in sorted_machines], fontsize=10)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Machines (Utilization %)', fontsize=12)
        ax.set_title(f'Phase 4 Machine Allocation - {scenario.replace("_", " ").title()}\nMachine Utilization View',
                    fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            patches.Patch(color=self.colors['late'], label='Late Jobs'),
            patches.Patch(color=self.colors['warning'], label='Warning (≤1 day)'),
            patches.Patch(color=self.colors['caution'], label='Caution (≤3 days)'),
            patches.Patch(color=self.colors['ok'], label='On Time')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Machine allocation chart saved: {save_path}")


def run_ppo_model_and_generate_schedule(model_path: str, env_class, max_steps: int = 200) -> Dict:
    """Run trained PPO model to generate a schedule."""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return {}
    
    try:
        # Load trained PPO model
        print(f"Loading PPO model: {model_path}")
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
        return schedule_data
        
    except Exception as e:
        print(f"Error running PPO model: {e}")
        import traceback
        traceback.print_exc()
        return {}


def discover_all_trained_models():
    """Discover all available trained models in phase4 results."""
    base_dir = '/Users/carrickcheah/Project/ppo/app_2/phase4/results'
    models = []
    
    # Check small_balanced models
    balanced_dir = os.path.join(base_dir, 'small_balanced')
    if os.path.exists(balanced_dir):
        # Final model
        final_model = os.path.join(balanced_dir, 'small_balanced_final.zip')
        if os.path.exists(final_model):
            models.append({
                'name': 'small_balanced_final',
                'scenario': 'small_balanced',
                'env_class': SmallBalancedEnvironment,
                'model_path': final_model
            })
        
        # Checkpoint models
        checkpoints_dir = os.path.join(balanced_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.zip')]
            # Test a few key checkpoints
            key_checkpoints = ['small_balanced_checkpoint_500000_steps.zip', 
                             'small_balanced_checkpoint_300000_steps.zip',
                             'small_balanced_checkpoint_100000_steps.zip']
            for checkpoint in key_checkpoints:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                if os.path.exists(checkpoint_path):
                    models.append({
                        'name': f'small_balanced_{checkpoint.replace(".zip", "")}',
                        'scenario': 'small_balanced',
                        'env_class': SmallBalancedEnvironment,
                        'model_path': checkpoint_path
                    })
    
    # Check small_rush models
    rush_dir = os.path.join(base_dir, 'small_rush')
    if os.path.exists(rush_dir):
        # Checkpoint models only (no final model yet)
        checkpoints_dir = os.path.join(rush_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            key_checkpoints = ['small_rush_checkpoint_300000_steps.zip',
                             'small_rush_checkpoint_200000_steps.zip',
                             'small_rush_checkpoint_100000_steps.zip']
            for checkpoint in key_checkpoints:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                if os.path.exists(checkpoint_path):
                    models.append({
                        'name': f'small_rush_{checkpoint.replace(".zip", "")}',
                        'scenario': 'small_rush', 
                        'env_class': SmallRushEnvironment,
                        'model_path': checkpoint_path
                    })
    
    return models


def main():
    """Main function to test ALL Phase 4 models and create comprehensive visualizations."""
    
    print("="*80)
    print("PHASE 4 COMPREHENSIVE PPO MODEL TESTING WITH GANTT VISUALIZATION")
    print("="*80)
    
    # Initialize visualizer
    visualizer = Phase4ScheduleVisualizer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Discover all available trained models
    print("Discovering all trained PPO models...")
    scenarios = discover_all_trained_models()
    print(f"Found {len(scenarios)} trained models to test")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario['name']} -> {scenario['model_path']}")
    
    # Define visualization directory and ensure it exists
    viz_dir = '/Users/carrickcheah/Project/ppo/app_2/visualizations/phase4'
    os.makedirs(viz_dir, exist_ok=True)
    
    results_summary = {
        'timestamp': timestamp,
        'total_models_tested': len(scenarios),
        'scenarios_tested': [],
        'visualizations_created': [],
        'performance_metrics': {
            'best_reward': -float('inf'),
            'best_model': None,
            'average_reward': 0,
            'models_compared': 0
        }
    }
    
    total_rewards = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Testing model: {scenario['name']}")
        print("-" * 70)
        print(f"Scenario: {scenario['scenario']}")
        print(f"Model: {scenario['model_path']}")
        
        # Verify model exists
        if not os.path.exists(scenario['model_path']):
            print(f"ERROR: Model not found: {scenario['model_path']}")
            continue
        
        # Run PPO model to generate schedule
        print("Running PPO model to generate schedule...")
        schedule_data = run_ppo_model_and_generate_schedule(
            scenario['model_path'], 
            scenario['env_class'],
            max_steps=500  # Increased for better scheduling
        )
        
        if not schedule_data:
            print(f"FAILED: Could not generate schedule for {scenario['name']}")
            continue
        
        reward = schedule_data.get('total_reward', 0)
        total_rewards.append(reward)
        
        # Track best performing model
        if reward > results_summary['performance_metrics']['best_reward']:
            results_summary['performance_metrics']['best_reward'] = reward
            results_summary['performance_metrics']['best_model'] = scenario['name']
        
        # Create visualizations with model-specific naming
        print("Creating Gantt chart visualizations...")
        
        # Job allocation chart
        job_chart_path = os.path.join(viz_dir, f"phase4_{scenario['name']}_job_allocation.png")
        visualizer.create_job_allocation_chart(schedule_data, job_chart_path, scenario['name'])
        
        # Machine allocation chart
        machine_chart_path = os.path.join(viz_dir, f"phase4_{scenario['name']}_machine_allocation.png")
        visualizer.create_machine_allocation_chart(schedule_data, machine_chart_path, scenario['name'])
        
        # Save schedule data
        schedule_path = os.path.join(viz_dir, f"phase4_{scenario['name']}_schedule.json")
        with open(schedule_path, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        # Update results
        results_summary['scenarios_tested'].append({
            'model_name': scenario['name'],
            'scenario': scenario['scenario'],
            'model_path': scenario['model_path'],
            'total_reward': reward,
            'steps_taken': schedule_data.get('steps_taken', 0),
            'scheduled_families': len(schedule_data.get('families', {})),
            'job_allocation_chart': job_chart_path,
            'machine_allocation_chart': machine_chart_path,
            'schedule_data': schedule_path
        })
        
        results_summary['visualizations_created'].extend([
            job_chart_path,
            machine_chart_path,
            schedule_path
        ])
        
        print(f"SUCCESS: {scenario['name']}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Steps: {schedule_data.get('steps_taken', 0)}")
        print(f"  Families: {len(schedule_data.get('families', {}))}")
        print(f"  Charts: job_allocation, machine_allocation")
    
    # Calculate performance metrics
    if total_rewards:
        results_summary['performance_metrics']['average_reward'] = sum(total_rewards) / len(total_rewards)
        results_summary['performance_metrics']['models_compared'] = len(total_rewards)
    
    # Save comprehensive summary
    summary_path = os.path.join(viz_dir, f"phase4_comprehensive_test_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create performance comparison visualization
    create_performance_comparison_chart(results_summary, viz_dir, timestamp)
    
    print("\n" + "="*80)
    print("PHASE 4 COMPREHENSIVE TESTING COMPLETED")
    print("="*80)
    print(f"Models tested: {len(results_summary['scenarios_tested'])}/{len(scenarios)}")
    print(f"Visualizations created: {len(results_summary['visualizations_created'])}")
    print(f"Best model: {results_summary['performance_metrics']['best_model']}")
    print(f"Best reward: {results_summary['performance_metrics']['best_reward']:.2f}")
    print(f"Average reward: {results_summary['performance_metrics']['average_reward']:.2f}")
    
    # Print results by scenario
    print("\nPERFORMANCE SUMMARY BY MODEL:")
    print("-" * 70)
    for result in sorted(results_summary['scenarios_tested'], key=lambda x: x['total_reward'], reverse=True):
        print(f"  {result['model_name']:<40} | Reward: {result['total_reward']:>8.2f} | Families: {result['scheduled_families']:>3}")
    
    print("\nGENERATED VISUALIZATIONS:")
    print("-" * 70)
    for result in results_summary['scenarios_tested']:
        print(f"\n{result['model_name']}:")
        print(f"  Job Allocation:     {result['job_allocation_chart']}")
        print(f"  Machine Allocation: {result['machine_allocation_chart']}")
        print(f"  Schedule Data:      {result['schedule_data']}")
    
    print(f"\nSummary Report: {summary_path}")
    
    return results_summary


def create_performance_comparison_chart(results_summary: Dict, viz_dir: str, timestamp: str):
    """Create a performance comparison chart across all tested models."""
    
    if not results_summary['scenarios_tested']:
        return
    
    # Extract data for plotting
    model_names = []
    rewards = []
    families_scheduled = []
    scenarios = []
    
    for result in sorted(results_summary['scenarios_tested'], key=lambda x: x['total_reward'], reverse=True):
        model_names.append(result['model_name'].replace('small_', '').replace('_checkpoint', '\nchkpt'))
        rewards.append(result['total_reward'])
        families_scheduled.append(result['scheduled_families'])
        scenarios.append(result['scenario'])
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Reward comparison
    colors = ['#1f77b4' if 'balanced' in s else '#ff7f0e' for s in scenarios]
    bars1 = ax1.bar(range(len(model_names)), rewards, color=colors, alpha=0.7)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Phase 4 PPO Models - Reward Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, reward in zip(bars1, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rewards)*0.01,
                f'{reward:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Families scheduled comparison
    bars2 = ax2.bar(range(len(model_names)), families_scheduled, color=colors, alpha=0.7)
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Families Scheduled', fontsize=12)
    ax2.set_title('Phase 4 PPO Models - Families Scheduled Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, families_scheduled):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(families_scheduled)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # Legend
    legend_elements = [
        patches.Patch(color='#1f77b4', alpha=0.7, label='Balanced Scenario'),
        patches.Patch(color='#ff7f0e', alpha=0.7, label='Rush Scenario')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save comparison chart
    comparison_chart_path = os.path.join(viz_dir, f"phase4_model_performance_comparison_{timestamp}.png")
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison chart saved: {comparison_chart_path}")
    results_summary['visualizations_created'].append(comparison_chart_path)


if __name__ == "__main__":
    main()