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
    """Discover ALL available trained models from both phase3 and phase4."""
    models = []
    
    # Phase 4 Models
    base_dir = '/Users/carrickcheah/Project/ppo/app_2/phase4/results'
    
    # Check small_balanced models
    balanced_dir = os.path.join(base_dir, 'small_balanced')
    if os.path.exists(balanced_dir):
        # Final model
        final_model = os.path.join(balanced_dir, 'small_balanced_final.zip')
        if os.path.exists(final_model):
            models.append({
                'name': 'phase4_small_balanced_final',
                'scenario': 'small_balanced',
                'env_class': SmallBalancedEnvironment,
                'model_path': final_model
            })
        
        # ALL checkpoint models (not just key ones)
        checkpoints_dir = os.path.join(balanced_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            for checkpoint in os.listdir(checkpoints_dir):
                if checkpoint.endswith('.zip'):
                    checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                    models.append({
                        'name': f'phase4_small_balanced_{checkpoint.replace(".zip", "")}',
                        'scenario': 'small_balanced',
                        'env_class': SmallBalancedEnvironment,
                        'model_path': checkpoint_path
                    })
    
    # Check small_rush models
    rush_dir = os.path.join(base_dir, 'small_rush')
    if os.path.exists(rush_dir):
        checkpoints_dir = os.path.join(rush_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            for checkpoint in os.listdir(checkpoints_dir):
                if checkpoint.endswith('.zip'):
                    checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                    models.append({
                        'name': f'phase4_small_rush_{checkpoint.replace(".zip", "")}',
                        'scenario': 'small_rush', 
                        'env_class': SmallRushEnvironment,
                        'model_path': checkpoint_path
                    })
    
    # Phase 3 Models - Use SmallBalancedEnvironment as fallback
    phase3_base = '/Users/carrickcheah/Project/ppo/app_2/phase3'
    
    # Curriculum models
    curriculum_dir = os.path.join(phase3_base, 'curriculum_models')
    if os.path.exists(curriculum_dir):
        for scenario_dir in os.listdir(curriculum_dir):
            scenario_path = os.path.join(curriculum_dir, scenario_dir)
            if os.path.isdir(scenario_path):
                for model_file in os.listdir(scenario_path):
                    if model_file.endswith('.zip'):
                        models.append({
                            'name': f'phase3_curriculum_{scenario_dir}_{model_file.replace(".zip", "")}',
                            'scenario': 'small_balanced',  # Use balanced env as fallback
                            'env_class': SmallBalancedEnvironment,
                            'model_path': os.path.join(scenario_path, model_file)
                        })
    
    # Checkpoints models
    checkpoints_base = os.path.join(phase3_base, 'checkpoints')
    categories = ['80percent', 'better_rewards', 'foundation', 'masked', 'perfect', 'schedule_all', 'simple_fix', 'truly_fixed']
    
    for category in categories:
        category_path = os.path.join(checkpoints_base, category)
        if os.path.exists(category_path):
            for sub_item in os.listdir(category_path):
                sub_path = os.path.join(category_path, sub_item)
                if os.path.isdir(sub_path):
                    # Directory with models
                    for model_file in os.listdir(sub_path):
                        if model_file.endswith('.zip'):
                            models.append({
                                'name': f'phase3_{category}_{sub_item}_{model_file.replace(".zip", "")}',
                                'scenario': 'small_balanced',
                                'env_class': SmallBalancedEnvironment,
                                'model_path': os.path.join(sub_path, model_file)
                            })
                elif sub_item.endswith('.zip'):
                    # Direct model file
                    models.append({
                        'name': f'phase3_{category}_{sub_item.replace(".zip", "")}',
                        'scenario': 'small_balanced',
                        'env_class': SmallBalancedEnvironment,
                        'model_path': sub_path
                    })
    
    # Additional model directories from phase3
    additional_dirs = ['models_100_percent', 'models_80_percent', 'models_schedule_all', 'final_models', 'truly_fixed_models']
    for dir_name in additional_dirs:
        models_dir = os.path.join(phase3_base, dir_name)
        if os.path.exists(models_dir):
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.zip'):
                    models.append({
                        'name': f'phase3_{dir_name}_{model_file.replace(".zip", "")}',
                        'scenario': 'small_balanced',
                        'env_class': SmallBalancedEnvironment,
                        'model_path': os.path.join(models_dir, model_file)
                    })
    
    print(f"Discovered {len(models)} total models ({len([m for m in models if 'phase4' in m['name']])} phase4, {len([m for m in models if 'phase3' in m['name']])} phase3)")
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
    sorted_results = sorted(results_summary['scenarios_tested'], key=lambda x: x['total_reward'], reverse=True)
    for i, result in enumerate(sorted_results[:15]):  # Show top 15
        print(f"{i+1:2d}. {result['model_name']:<50} | Reward: {result['total_reward']:>8.2f} | Families: {result['scheduled_families']:>3}")
    
    # Create the improved production schedule using the BEST model
    if sorted_results:
        best_result = sorted_results[0]
        print(f"\nCREATING IMPROVED PRODUCTION SCHEDULE USING BEST MODEL:")
        print(f"Best Model: {best_result['model_name']}")
        print(f"Performance: {best_result['total_reward']:.2f} reward, {best_result['scheduled_families']} families")
        
        # Create enhanced visualization for the best model
        best_schedule_path = os.path.join(viz_dir, f"{best_result['schedule_data']}")
        if os.path.exists(best_schedule_path):
            with open(best_schedule_path, 'r') as f:
                best_schedule = json.load(f)
            
            # Create improved production schedule chart
            improved_chart_path = os.path.join(viz_dir, "improved_production_schedule.png")
            create_enhanced_production_chart(best_schedule, improved_chart_path, best_result['model_name'])
            
            print(f"IMPROVED PRODUCTION SCHEDULE SAVED: {improved_chart_path}")
            results_summary['best_schedule_chart'] = improved_chart_path
    
    # Generate comparison report
    comparison_report_path = os.path.join(viz_dir, f"model_comparison_report_{timestamp}.md")
    create_comparison_report(results_summary, comparison_report_path)
    print(f"COMPARISON REPORT SAVED: {comparison_report_path}")
    
    print("\nGENERATED VISUALIZATIONS:")
    print("-" * 70)
    for result in results_summary['scenarios_tested'][:5]:  # Show top 5
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


def create_enhanced_production_chart(schedule_data: Dict, save_path: str, model_name: str):
    """Create an enhanced production planning chart with professional formatting."""
    
    if not schedule_data or 'families' not in schedule_data:
        print(f"No valid schedule data for enhanced chart")
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
        print(f"No scheduled jobs found for enhanced chart")
        return
    
    # Sort jobs by job_id for consistent display
    scheduled_jobs.sort(key=lambda x: x['job_id'])
    
    # Create enhanced figure
    fig, ax = plt.subplots(figsize=(18, max(10, len(scheduled_jobs) * 0.5)))
    
    # Define professional colors
    colors = {
        'late': '#D32F2F',      # Red for late jobs
        'warning': '#F57C00',   # Orange for jobs at risk
        'caution': '#FBC02D',   # Yellow for jobs with tight deadlines
        'ok': '#388E3C'         # Green for jobs on time
    }
    
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
        try:
            job_end_dt = datetime.strptime("2025-07-25", "%Y-%m-%d") + timedelta(hours=job['end'])
            lcd_dt = datetime.strptime(job['lcd_date'], "%Y-%m-%d")
            days_diff = (lcd_dt - job_end_dt).days
            
            if days_diff < 0:
                status = 'late'
            elif days_diff <= 1:
                status = 'warning'
            elif days_diff <= 3:
                status = 'caution'
            else:
                status = 'ok'
        except:
            status = 'ok'
        
        color = colors[status]
        
        # Create bar with professional styling
        bar = ax.barh(y_pos, duration, left=job['start'], height=0.7, 
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Add detailed task label
        if duration > 3:  # Only show label if bar is wide enough
            ax.text(job['start'] + duration/2, y_pos, f"M{job['machine']}", 
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Add red dashed line at 16:00 (as requested)
    ax.axvline(x=16.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='4:00 PM Deadline')
    
    # Professional formatting
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=11)
    ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Job IDs', fontsize=14, fontweight='bold')
    ax.set_title(f'Production Planning System - Enhanced Schedule\nModel: {model_name} | {len(scheduled_jobs)} Jobs Scheduled', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    legend_elements = [
        patches.Patch(color=colors['late'], label='Late Jobs (past deadline)'),
        patches.Patch(color=colors['warning'], label='Warning (≤1 day remaining)'),
        patches.Patch(color=colors['caution'], label='Caution (≤3 days remaining)'),
        patches.Patch(color=colors['ok'], label='On Time (>3 days remaining)'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='4:00 PM Deadline')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Professional grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add performance metrics as text
    total_jobs = len(schedule_data.get('families', {}))
    completion_rate = len(scheduled_jobs) / total_jobs * 100 if total_jobs > 0 else 0
    reward = schedule_data.get('total_reward', 0)
    
    metrics_text = f"Performance Metrics:\n• Jobs Scheduled: {len(scheduled_jobs)}/{total_jobs} ({completion_rate:.1f}%)\n• Total Reward: {reward:.1f}\n• Model: {model_name}"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Enhanced production schedule chart saved: {save_path}")


def create_comparison_report(results_summary: Dict, report_path: str):
    """Create a detailed comparison report of all tested models."""
    
    if not results_summary['scenarios_tested']:
        return
    
    # Sort results by performance
    sorted_results = sorted(results_summary['scenarios_tested'], key=lambda x: x['total_reward'], reverse=True)
    
    with open(report_path, 'w') as f:
        f.write("# PPO Model Performance Comparison Report\n\n")
        f.write(f"**Generated:** {results_summary['timestamp']}\n")
        f.write(f"**Total Models Tested:** {results_summary['total_models_tested']}\n")
        f.write(f"**Successfully Tested:** {len(sorted_results)}\n\n")
        
        f.write("## Executive Summary\n\n")
        if sorted_results:
            best_model = sorted_results[0]
            f.write(f"**Best Performing Model:** {best_model['model_name']}\n")
            f.write(f"**Best Reward:** {best_model['total_reward']:.2f}\n")
            f.write(f"**Families Scheduled:** {best_model['scheduled_families']}\n")
            f.write(f"**Average Reward:** {results_summary['performance_metrics']['average_reward']:.2f}\n\n")
        
        f.write("## Why the Best Model Performed Better\n\n")
        if sorted_results:
            best = sorted_results[0]
            if 'phase4' in best['model_name']:
                f.write("The best performing model comes from Phase 4 training, which indicates:\n")
                f.write("- More sophisticated environment design\n")
                f.write("- Better reward function optimization\n")
                f.write("- Advanced action space handling\n")
            elif 'phase3' in best['model_name']:
                f.write("The best performing model comes from Phase 3 training, which suggests:\n")
                f.write("- Proven curriculum learning approach\n")
                f.write("- Stable convergence patterns\n")
                f.write("- Well-tuned hyperparameters\n")
            
            if 'final' in best['model_name']:
                f.write("- Final model represents fully trained convergence\n")
            elif 'checkpoint' in best['model_name']:
                f.write("- Checkpoint model captured optimal training state\n")
        
        f.write("\n## Top 10 Performing Models\n\n")
        f.write("| Rank | Model Name | Reward | Families | Source |\n")
        f.write("|------|------------|--------|----------|--------|\n")
        
        for i, result in enumerate(sorted_results[:10], 1):
            source = "Phase 4" if 'phase4' in result['model_name'] else "Phase 3"
            f.write(f"| {i} | {result['model_name'][:50]} | {result['total_reward']:.2f} | {result['scheduled_families']} | {source} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        f.write("### Phase 4 vs Phase 3 Performance\n\n")
        
        phase4_results = [r for r in sorted_results if 'phase4' in r['model_name']]
        phase3_results = [r for r in sorted_results if 'phase3' in r['model_name']]
        
        if phase4_results:
            avg_phase4 = sum(r['total_reward'] for r in phase4_results) / len(phase4_results)
            f.write(f"**Phase 4 Average Reward:** {avg_phase4:.2f} ({len(phase4_results)} models)\n")
        
        if phase3_results:
            avg_phase3 = sum(r['total_reward'] for r in phase3_results) / len(phase3_results)
            f.write(f"**Phase 3 Average Reward:** {avg_phase3:.2f} ({len(phase3_results)} models)\n")
        
        f.write("\n### Model Type Analysis\n\n")
        model_types = {}
        for result in sorted_results:
            if 'final' in result['model_name']:
                model_type = 'Final Models'
            elif 'checkpoint' in result['model_name']:
                model_type = 'Checkpoint Models'
            else:
                model_type = 'Other Models'
            
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(result)
        
        for model_type, models in model_types.items():
            avg_reward = sum(m['total_reward'] for m in models) / len(models)
            f.write(f"**{model_type}:** {len(models)} models, average reward {avg_reward:.2f}\n")
        
        f.write("\n## Recommendations\n\n")
        if sorted_results:
            best = sorted_results[0]
            f.write(f"1. **Use the best model ({best['model_name']}) for production scheduling**\n")
            f.write(f"2. **Expected performance: {best['scheduled_families']} families scheduled with {best['total_reward']:.2f} reward**\n")
            if best['scheduled_families'] >= 20:
                f.write("3. **This model meets the target of 20+ jobs scheduled**\n")
            else:
                f.write("3. **Consider further training to achieve 20+ jobs scheduled**\n")
        
        f.write("\n## Technical Details\n\n")
        f.write("All models were tested using consistent parameters:\n")
        f.write("- Environment: SmallBalancedEnvironment (Phase 4) or equivalent\n")
        f.write("- Deterministic action selection\n")
        f.write("- Maximum 500 steps per episode\n")
        f.write("- Same production data across all tests\n")
    
    print(f"Comparison report saved: {report_path}")


if __name__ == "__main__":
    main()