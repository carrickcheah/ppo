"""
Evaluation Module for PPO Scheduling Model

Provides comprehensive evaluation capabilities:
- Performance metrics (makespan, tardiness, utilization)
- Visualization of schedules (Gantt charts)
- Comparison with baseline algorithms
- Statistical analysis of results
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.scheduling_game_env import SchedulingGameEnv
from src.data.data_loader import DataLoader
from phase2.ppo_scheduler import PPOScheduler

logger = logging.getLogger(__name__)


class SchedulingEvaluator:
    """
    Comprehensive evaluation of scheduling models.
    
    Provides metrics, visualizations, and comparisons for
    trained PPO models and baseline algorithms.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Metrics to track
        self.metrics_names = [
            'makespan',
            'total_tardiness',
            'mean_tardiness',
            'on_time_rate',
            'machine_utilization',
            'mean_flow_time',
            'max_flow_time',
            'setup_time_ratio',
            'episode_reward'
        ]
        
    def evaluate_model(self,
                      model: PPOScheduler,
                      env: SchedulingGameEnv,
                      n_episodes: int = 10,
                      deterministic: bool = True,
                      render: bool = False) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained PPO model
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy
            render: Whether to render episodes
            
        Returns:
            Dictionary of aggregated metrics
        """
        model.eval()
        
        episode_metrics = []
        episode_schedules = []
        
        for episode in range(n_episodes):
            metrics, schedule = self.run_episode(
                model, env, deterministic, render
            )
            episode_metrics.append(metrics)
            episode_schedules.append(schedule)
            
        # Aggregate metrics
        aggregated = self._aggregate_metrics(episode_metrics)
        
        # Store schedules for visualization
        self.last_schedules = episode_schedules
        
        return aggregated
    
    def run_episode(self,
                   model: PPOScheduler,
                   env: SchedulingGameEnv,
                   deterministic: bool = True,
                   render: bool = False) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Run a single evaluation episode.
        
        Args:
            model: Model to evaluate
            env: Environment
            deterministic: Use deterministic policy
            render: Whether to render
            
        Returns:
            metrics: Episode metrics
            schedule: Generated schedule
        """
        obs, info = env.reset()
        done = False
        
        n_jobs = len(env.jobs)
        n_machines = len(env.machines)
        
        while not done:
            # Get action from model
            action_mask = env.get_action_mask()
            action, _, _ = model.get_action(
                obs, action_mask, deterministic,
                n_jobs=n_jobs, n_machines=n_machines
            )
            
            # Decode action
            job_idx = action // n_machines
            machine_idx = action % n_machines
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(
                np.array([job_idx, machine_idx])
            )
            done = terminated or truncated
            
            if render:
                env.render()
                
        # Get final metrics
        metrics = self._extract_metrics(env, info)
        
        # Get final schedule
        schedule = self._extract_schedule(env)
        
        return metrics, schedule
    
    def _extract_metrics(self, 
                        env: SchedulingGameEnv,
                        info: Dict) -> Dict[str, float]:
        """Extract metrics from completed episode."""
        metrics = {}
        
        # Basic metrics from info
        metrics['episode_reward'] = info.get('episode_reward', 0)
        metrics['makespan'] = env.get_current_time()
        
        # Tardiness metrics
        tardiness_list = []
        on_time_count = 0
        
        for job in env.jobs:
            if job.get('scheduled'):
                completion_time = job.get('completion_time', 0)
                due_date = job.get('TargetDate_dd', float('inf'))
                
                # Convert due date if it's a datetime
                if isinstance(due_date, (datetime, pd.Timestamp)):
                    # Convert to hours from start
                    due_date_hours = (due_date - env.start_time).total_seconds() / 3600
                else:
                    due_date_hours = due_date * 24  # Days to hours
                    
                tardiness = max(0, completion_time - due_date_hours)
                tardiness_list.append(tardiness)
                
                if tardiness == 0:
                    on_time_count += 1
                    
        metrics['total_tardiness'] = sum(tardiness_list)
        metrics['mean_tardiness'] = np.mean(tardiness_list) if tardiness_list else 0
        metrics['on_time_rate'] = on_time_count / len(env.jobs) if env.jobs else 0
        
        # Machine utilization
        total_busy_time = 0
        for machine in env.machines:
            busy_time = sum(task['duration'] for task in machine.get('schedule', []))
            total_busy_time += busy_time
            
        total_available_time = metrics['makespan'] * len(env.machines)
        metrics['machine_utilization'] = total_busy_time / total_available_time if total_available_time > 0 else 0
        
        # Flow time metrics
        flow_times = []
        for job in env.jobs:
            if job.get('scheduled'):
                # Flow time = completion time - arrival time (assuming all arrive at 0)
                flow_times.append(job.get('completion_time', 0))
                
        metrics['mean_flow_time'] = np.mean(flow_times) if flow_times else 0
        metrics['max_flow_time'] = max(flow_times) if flow_times else 0
        
        # Setup time ratio
        total_setup_time = sum(job.get('SetupTime_d', 0) / 60 for job in env.jobs if job.get('scheduled'))
        total_processing_time = sum(job.get('processing_time', 0) for job in env.jobs if job.get('scheduled'))
        metrics['setup_time_ratio'] = total_setup_time / total_processing_time if total_processing_time > 0 else 0
        
        return metrics
    
    def _extract_schedule(self, env: SchedulingGameEnv) -> List[Dict]:
        """Extract schedule information for visualization."""
        schedule = []
        
        for job in env.jobs:
            if job.get('scheduled'):
                schedule_entry = {
                    'job_id': job['JoId_v'],
                    'task': job.get('Task_v', 'Unknown'),
                    'machine_id': job.get('scheduled_machine_id'),
                    'machine_name': job.get('scheduled_machine_name', 'Unknown'),
                    'start_time': job.get('start_time', 0),
                    'end_time': job.get('completion_time', 0),
                    'duration': job.get('processing_time', 0),
                    'setup_time': job.get('SetupTime_d', 0) / 60,
                    'due_date': job.get('TargetDate_dd', None),
                    'is_important': job.get('IsImportant', 0),
                    'required_machines': job.get('required_machines', [])
                }
                schedule.append(schedule_entry)
                
        return schedule
    
    def _aggregate_metrics(self, 
                          episode_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across episodes."""
        aggregated = {}
        
        for metric in self.metrics_names:
            values = [ep.get(metric, 0) for ep in episode_metrics]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            
        return aggregated
    
    def visualize_schedule(self,
                          schedule: List[Dict],
                          save_path: Optional[str] = None,
                          title: str = "Schedule Gantt Chart"):
        """
        Create Gantt chart visualization of schedule.
        
        Args:
            schedule: Schedule to visualize
            save_path: Path to save figure
            title: Chart title
        """
        if not schedule:
            logger.warning("Empty schedule, nothing to visualize")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get unique machines and assign colors
        machines = sorted(list(set(s['machine_name'] for s in schedule)))
        machine_to_y = {m: i for i, m in enumerate(machines)}
        
        # Color map for jobs
        job_colors = plt.cm.Set3(np.linspace(0, 1, len(schedule)))
        
        # Plot jobs
        for i, job in enumerate(schedule):
            y_pos = machine_to_y[job['machine_name']]
            
            # Main processing block
            rect = patches.Rectangle(
                (job['start_time'], y_pos - 0.4),
                job['duration'],
                0.8,
                linewidth=1,
                edgecolor='black',
                facecolor=job_colors[i],
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add job label
            ax.text(
                job['start_time'] + job['duration'] / 2,
                y_pos,
                f"{job['job_id']}\n{job['task']}",
                ha='center',
                va='center',
                fontsize=8,
                weight='bold' if job['is_important'] else 'normal'
            )
            
            # Add setup time if significant
            if job['setup_time'] > 0:
                setup_rect = patches.Rectangle(
                    (job['start_time'] - job['setup_time'], y_pos - 0.4),
                    job['setup_time'],
                    0.8,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='gray',
                    alpha=0.5,
                    hatch='//'
                )
                ax.add_patch(setup_rect)
                
        # Add due date lines
        for job in schedule:
            if job['due_date'] is not None:
                due_hours = job['due_date'] * 24 if isinstance(job['due_date'], (int, float)) else 0
                if due_hours > 0:
                    ax.axvline(
                        x=due_hours,
                        color='red',
                        linestyle='--',
                        alpha=0.3,
                        linewidth=1
                    )
                    
        # Formatting
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels(machines)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Machine', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Set x-axis limits
        max_time = max(job['end_time'] for job in schedule)
        ax.set_xlim(0, max_time * 1.05)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='gray', alpha=0.5, hatch='//', label='Setup Time'),
            patches.Patch(facecolor='white', edgecolor='black', label='Processing Time'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Due Date')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Schedule visualization saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def compare_algorithms(self,
                          algorithms: Dict[str, Any],
                          test_data: Dict,
                          n_episodes: int = 10) -> pd.DataFrame:
        """
        Compare multiple algorithms on same test data.
        
        Args:
            algorithms: Dict mapping algorithm names to models/functions
            test_data: Test data configuration
            n_episodes: Episodes per algorithm
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for algo_name, algo_model in algorithms.items():
            logger.info(f"Evaluating {algo_name}...")
            
            # Create fresh environment for each algorithm
            env = self._create_test_environment(test_data)
            
            # Evaluate
            if hasattr(algo_model, 'get_action'):
                # It's a trained model
                metrics = self.evaluate_model(
                    algo_model, env, n_episodes, deterministic=True
                )
            else:
                # It's a baseline function
                metrics = self._evaluate_baseline(
                    algo_model, env, n_episodes
                )
                
            # Add algorithm name
            metrics['algorithm'] = algo_name
            results.append(metrics)
            
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['algorithm'] + [col for col in df.columns if col != 'algorithm']
        df = df[cols]
        
        return df
    
    def _create_test_environment(self, test_data: Dict) -> SchedulingGameEnv:
        """Create test environment from configuration."""
        # Load data
        data_loader = DataLoader(test_data['data_config'])
        jobs = data_loader.load_jobs()
        machines = data_loader.load_machines()
        working_hours = data_loader.load_working_hours()
        
        # Create environment
        env = SchedulingGameEnv(
            jobs=jobs,
            machines=machines,
            working_hours=working_hours,
            config=test_data.get('env_config', {})
        )
        
        return env
    
    def _evaluate_baseline(self,
                          baseline_fn,
                          env: SchedulingGameEnv,
                          n_episodes: int) -> Dict[str, float]:
        """Evaluate a baseline algorithm."""
        episode_metrics = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                # Get action from baseline
                action = baseline_fn(env)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            # Get metrics
            metrics = self._extract_metrics(env, info)
            episode_metrics.append(metrics)
            
        # Aggregate
        return self._aggregate_metrics(episode_metrics)
    
    def generate_report(self,
                       evaluation_results: Dict,
                       save_path: str):
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluation
            save_path: Path to save report
        """
        report = []
        report.append("# PPO Scheduling Model Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Model Performance
        report.append("## Model Performance Metrics\n")
        for metric, value in evaluation_results.items():
            if '_mean' in metric:
                base_metric = metric.replace('_mean', '')
                mean_val = value
                std_val = evaluation_results.get(f'{base_metric}_std', 0)
                report.append(f"- **{base_metric}**: {mean_val:.4f} ± {std_val:.4f}")
                
        # Key insights
        report.append("\n## Key Insights\n")
        
        makespan = evaluation_results.get('makespan_mean', 0)
        utilization = evaluation_results.get('machine_utilization_mean', 0)
        on_time = evaluation_results.get('on_time_rate_mean', 0)
        
        report.append(f"- Average makespan: {makespan:.2f} hours")
        report.append(f"- Machine utilization: {utilization:.2%}")
        report.append(f"- On-time delivery rate: {on_time:.2%}")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        
        if utilization < 0.7:
            report.append("- Low machine utilization detected. Consider:")
            report.append("  - Reducing setup times through better job sequencing")
            report.append("  - Batching similar jobs together")
            
        if on_time < 0.9:
            report.append("- On-time rate below target. Suggest:")
            report.append("  - Prioritizing jobs with tight deadlines")
            report.append("  - Adding buffer time for critical jobs")
            
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
            
        logger.info(f"Evaluation report saved to {save_path}")


def create_baseline_algorithms():
    """Create baseline scheduling algorithms for comparison."""
    
    def fifo_scheduler(env):
        """First In First Out scheduling."""
        # Get unscheduled jobs
        unscheduled = [i for i, job in enumerate(env.jobs) 
                      if not job.get('scheduled', False)]
        
        if not unscheduled:
            return np.array([0, 0])  # Dummy action
            
        # Select first unscheduled job
        job_idx = unscheduled[0]
        
        # Find first available machine
        valid_machines = env._get_valid_machines_for_job(job_idx)
        if valid_machines:
            machine_idx = valid_machines[0]
            return np.array([job_idx, machine_idx])
        else:
            return np.array([0, 0])
            
    def edd_scheduler(env):
        """Earliest Due Date scheduling."""
        # Get unscheduled jobs with due dates
        unscheduled_with_dd = []
        for i, job in enumerate(env.jobs):
            if not job.get('scheduled', False):
                due_date = job.get('TargetDate_dd', float('inf'))
                unscheduled_with_dd.append((i, due_date))
                
        if not unscheduled_with_dd:
            return np.array([0, 0])
            
        # Sort by due date
        unscheduled_with_dd.sort(key=lambda x: x[1])
        
        # Try jobs in order
        for job_idx, _ in unscheduled_with_dd:
            valid_machines = env._get_valid_machines_for_job(job_idx)
            if valid_machines:
                machine_idx = valid_machines[0]
                return np.array([job_idx, machine_idx])
                
        return np.array([0, 0])
        
    def spt_scheduler(env):
        """Shortest Processing Time scheduling."""
        # Get unscheduled jobs with processing times
        unscheduled_with_pt = []
        for i, job in enumerate(env.jobs):
            if not job.get('scheduled', False):
                proc_time = job.get('processing_time', float('inf'))
                unscheduled_with_pt.append((i, proc_time))
                
        if not unscheduled_with_pt:
            return np.array([0, 0])
            
        # Sort by processing time
        unscheduled_with_pt.sort(key=lambda x: x[1])
        
        # Try jobs in order
        for job_idx, _ in unscheduled_with_pt:
            valid_machines = env._get_valid_machines_for_job(job_idx)
            if valid_machines:
                machine_idx = valid_machines[0]
                return np.array([job_idx, machine_idx])
                
        return np.array([0, 0])
        
    return {
        'FIFO': fifo_scheduler,
        'EDD': edd_scheduler,
        'SPT': spt_scheduler
    }


def main():
    """Main entry point for evaluation."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Evaluate PPO scheduling model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml',
                       help='Path to evaluation configuration')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with baseline algorithms')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate schedule visualizations')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/',
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configurations
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)
        
    # Create evaluator
    evaluator = SchedulingEvaluator(eval_config)
    
    # Load model
    model_config_path = args.config.replace('evaluation', 'training')
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    model = PPOScheduler(model_config)
    model.load(args.model)
    
    # Create test environment
    test_env = evaluator._create_test_environment(eval_config['test_data'])
    
    # Evaluate model
    logger.info("Evaluating PPO model...")
    results = evaluator.evaluate_model(
        model, test_env, args.n_episodes, deterministic=True
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    evaluator.generate_report(results, report_path)
    
    # Visualize schedule
    if args.visualize and evaluator.last_schedules:
        for i, schedule in enumerate(evaluator.last_schedules[:3]):  # First 3
            viz_path = os.path.join(args.output_dir, f'schedule_episode_{i}.png')
            evaluator.visualize_schedule(
                schedule, viz_path, 
                title=f"PPO Generated Schedule - Episode {i+1}"
            )
            
    # Compare with baselines
    if args.compare:
        logger.info("Comparing with baseline algorithms...")
        
        baselines = create_baseline_algorithms()
        algorithms = {'PPO': model}
        algorithms.update(baselines)
        
        comparison_df = evaluator.compare_algorithms(
            algorithms, eval_config['test_data'], n_episodes=5
        )
        
        # Save comparison
        comparison_path = os.path.join(args.output_dir, 'algorithm_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison saved to {comparison_path}")
        
        # Print summary
        print("\nAlgorithm Comparison Summary:")
        print(comparison_df[['algorithm', 'makespan_mean', 'on_time_rate_mean', 
                           'machine_utilization_mean']].to_string(index=False))
        

if __name__ == '__main__':
    main()