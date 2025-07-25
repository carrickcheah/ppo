"""
Phase 3 Evaluation and Visualization Tools
Generates Gantt charts and performance reports from trained models
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScheduleVisualizer:
    """Visualizes schedules from trained PPO models."""
    
    def __init__(self, 
                 checkpoint_dir: str = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints",
                 visualization_dir: str = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"):
        self.checkpoint_dir = checkpoint_dir
        self.visualization_dir = visualization_dir
        os.makedirs(self.visualization_dir, exist_ok=True)
        
    def load_model_and_env(self, stage_name: str) -> Tuple[PPO, VecNormalize]:
        """Load trained model and environment for a stage."""
        # Model path
        model_path = os.path.join(self.checkpoint_dir, stage_name, "final_model.zip")
        if not os.path.exists(model_path):
            # Try best model
            model_path = os.path.join(self.checkpoint_dir, stage_name, "best", "best_model.zip")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for stage {stage_name}")
        
        # Load model
        model = PPO.load(model_path)
        
        # Create environment
        def make_env():
            env = CurriculumEnvironmentReal(stage_name=stage_name, verbose=False)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Load normalization stats if available
        vec_norm_path = os.path.join(self.checkpoint_dir, stage_name, "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, training=False, norm_reward=False)
        
        return model, env
    
    def generate_schedule(self, model: PPO, env: VecNormalize) -> Dict:
        """Generate a complete schedule using the trained model."""
        obs = env.reset()
        done = False
        
        schedule = {
            'jobs': [],
            'machines': {},
            'metrics': {
                'total_jobs': 0,
                'scheduled_jobs': 0,
                'on_time_jobs': 0,
                'late_jobs': 0,
                'utilization': 0.0,
                'makespan': 0.0
            }
        }
        
        # Get environment info - unwrap properly
        base_env = env.envs[0].env if hasattr(env.envs[0], 'env') else env.envs[0]
        
        # Initialize machine schedules
        for machine_id in base_env.machine_ids:
            schedule['machines'][machine_id] = []
        
        steps = 0
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Extract final schedule from environment
        for job_key, assignment in base_env.job_assignments.items():
            job_info = {
                'job_id': job_key,
                'machines': assignment['machines'],
                'start_time': assignment['start'],
                'end_time': assignment['end'],
                'duration': assignment['end'] - assignment['start']
            }
            schedule['jobs'].append(job_info)
            
            # Add to machine schedules
            for machine_id in assignment['machines']:
                schedule['machines'][machine_id].append({
                    'job': job_key,
                    'start': assignment['start'],
                    'end': assignment['end']
                })
        
        # Calculate metrics
        schedule['metrics']['total_jobs'] = len(base_env.families)
        schedule['metrics']['scheduled_jobs'] = len(base_env.completed_jobs)
        
        # Calculate utilization and makespan
        if schedule['jobs']:
            makespan = max(job['end_time'] for job in schedule['jobs'])
            schedule['metrics']['makespan'] = makespan
            
            # Calculate utilization
            total_busy_time = 0
            for machine_id, jobs in schedule['machines'].items():
                for job in jobs:
                    total_busy_time += job['end'] - job['start']
            
            if makespan > 0:
                total_capacity = makespan * len(base_env.machine_ids)
                schedule['metrics']['utilization'] = total_busy_time / total_capacity
        
        return schedule
    
    def create_gantt_chart(self, schedule: Dict, stage_name: str, save_path: Optional[str] = None):
        """Create a Gantt chart visualization of the schedule."""
        if not schedule['jobs']:
            logger.warning("No jobs to visualize")
            return
        
        # Sort machines for consistent display
        machine_ids = sorted(schedule['machines'].keys())
        machine_positions = {m_id: idx for idx, m_id in enumerate(machine_ids)}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(8, len(machine_ids) * 0.3)))
        
        # Color map for jobs
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        job_colors = {}
        
        # Plot jobs
        for job_info in schedule['jobs']:
            job_id = job_info['job_id']
            
            # Extract family ID for coloring
            family_id = job_id.split('_seq')[0]
            if family_id not in job_colors:
                job_colors[family_id] = colors[len(job_colors) % len(colors)]
            
            color = job_colors[family_id]
            
            # Plot on each machine used
            for machine_id in job_info['machines']:
                y_pos = machine_positions[machine_id]
                
                # Create rectangle
                rect = mpatches.Rectangle(
                    (job_info['start_time'], y_pos - 0.4),
                    job_info['duration'],
                    0.8,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(rect)
                
                # Add job label if space allows
                if job_info['duration'] > 2:
                    ax.text(
                        job_info['start_time'] + job_info['duration'] / 2,
                        y_pos,
                        job_id.split('_')[-1],  # Just show sequence number
                        ha='center',
                        va='center',
                        fontsize=8,
                        weight='bold'
                    )
        
        # Set labels and title
        ax.set_yticks(range(len(machine_ids)))
        ax.set_yticklabels([f"M{m_id}" for m_id in machine_ids])
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_title(f'Schedule Gantt Chart - Stage: {stage_name}', fontsize=14, weight='bold')
        
        # Set limits
        ax.set_xlim(0, schedule['metrics']['makespan'] * 1.05)
        ax.set_ylim(-0.5, len(machine_ids) - 0.5)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add metrics text
        metrics_text = (
            f"Scheduled: {schedule['metrics']['scheduled_jobs']}/{schedule['metrics']['total_jobs']} jobs\n"
            f"Utilization: {schedule['metrics']['utilization']:.1%}\n"
            f"Makespan: {schedule['metrics']['makespan']:.1f} hours"
        )
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10
        )
        
        # Add legend for job families
        if len(job_colors) <= 10:
            legend_elements = [
                mpatches.Patch(color=color, label=family_id)
                for family_id, color in job_colors.items()
            ]
            ax.legend(
                handles=legend_elements,
                loc='upper right',
                bbox_to_anchor=(1.15, 1),
                title='Job Families'
            )
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(
                self.visualization_dir,
                f"gantt_chart_{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gantt chart saved to: {save_path}")
        return save_path
    
    def evaluate_stage(self, stage_name: str, n_episodes: int = 5) -> Dict:
        """Evaluate a trained model on its stage."""
        logger.info(f"\nEvaluating stage: {stage_name}")
        
        try:
            model, env = self.load_model_and_env(stage_name)
        except FileNotFoundError as e:
            logger.error(f"Cannot evaluate {stage_name}: {e}")
            return {}
        
        # Run multiple episodes
        episode_metrics = []
        
        for episode in range(n_episodes):
            schedule = self.generate_schedule(model, env)
            
            # Create Gantt chart for first episode
            if episode == 0:
                gantt_path = os.path.join(
                    self.visualization_dir,
                    f"gantt_{stage_name}.png"
                )
                self.create_gantt_chart(schedule, stage_name, gantt_path)
            
            episode_metrics.append(schedule['metrics'])
        
        # Aggregate metrics
        aggregated = {
            'stage_name': stage_name,
            'n_episodes': n_episodes,
            'utilization': {
                'mean': np.mean([m['utilization'] for m in episode_metrics]),
                'std': np.std([m['utilization'] for m in episode_metrics])
            },
            'completion_rate': {
                'mean': np.mean([m['scheduled_jobs'] / m['total_jobs'] for m in episode_metrics]),
                'std': np.std([m['scheduled_jobs'] / m['total_jobs'] for m in episode_metrics])
            },
            'makespan': {
                'mean': np.mean([m['makespan'] for m in episode_metrics]),
                'std': np.std([m['makespan'] for m in episode_metrics])
            }
        }
        
        logger.info(f"  Utilization: {aggregated['utilization']['mean']:.1%} ± {aggregated['utilization']['std']:.1%}")
        logger.info(f"  Completion: {aggregated['completion_rate']['mean']:.1%} ± {aggregated['completion_rate']['std']:.1%}")
        logger.info(f"  Makespan: {aggregated['makespan']['mean']:.1f} ± {aggregated['makespan']['std']:.1f} hours")
        
        return aggregated
    
    def create_training_progress_chart(self, log_dir: str = "/Users/carrickcheah/Project/ppo/app_2/phase3/logs"):
        """Create a chart showing training progress across stages."""
        stages_data = []
        
        # Load metrics for each stage
        for stage_file in sorted(os.listdir(log_dir)):
            if stage_file.startswith("stage_") and stage_file.endswith("_metrics.json"):
                with open(os.path.join(log_dir, stage_file), 'r') as f:
                    data = json.load(f)
                    stages_data.append(data)
        
        if not stages_data:
            logger.warning("No training metrics found")
            return
        
        # Sort by stage order
        stage_order = [
            'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
            'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
            'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
            'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
        ]
        
        stages_data.sort(key=lambda x: stage_order.index(x['stage_name']) if x['stage_name'] in stage_order else 999)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Mean rewards across stages
        stage_names = [s['stage_name'] for s in stages_data]
        mean_rewards = [s['mean_reward'] for s in stages_data]
        std_rewards = [s['std_reward'] for s in stages_data]
        
        x_pos = np.arange(len(stage_names))
        ax1.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, color='skyblue', edgecolor='navy')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stage_names, rotation=45, ha='right')
        ax1.set_ylabel('Mean Episode Reward', fontsize=12)
        ax1.set_title('Training Rewards Across Curriculum Stages', fontsize=14, weight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add phase separators
        phase_boundaries = [4, 8, 12, 16]
        phase_labels = ['Foundation', 'Strategy', 'Scale', 'Production']
        for i, boundary in enumerate(phase_boundaries[:-1]):
            ax1.axvline(x=boundary - 0.5, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Episode lengths
        mean_lengths = [s['mean_length'] for s in stages_data]
        ax2.plot(x_pos, mean_lengths, marker='o', markersize=8, linewidth=2, color='green')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stage_names, rotation=45, ha='right')
        ax2.set_ylabel('Mean Episode Length', fontsize=12)
        ax2.set_title('Episode Lengths Across Stages', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.visualization_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress chart saved to: {save_path}")


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and visualize trained models")
    parser.add_argument(
        '--stage',
        type=str,
        help='Specific stage to evaluate (e.g., toy_easy, small_rush)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluate all available stages'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=5,
        help='Number of episodes to evaluate per stage'
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Generate training progress chart'
    )
    
    args = parser.parse_args()
    
    visualizer = ScheduleVisualizer()
    
    if args.progress:
        visualizer.create_training_progress_chart()
    
    if args.stage:
        # Evaluate specific stage
        visualizer.evaluate_stage(args.stage, n_episodes=args.n_episodes)
    
    elif args.all:
        # Evaluate all stages
        stage_order = [
            'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
            'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
            'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
            'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
        ]
        
        all_results = []
        for stage in stage_order:
            result = visualizer.evaluate_stage(stage, n_episodes=args.n_episodes)
            if result:
                all_results.append(result)
        
        # Save summary
        if all_results:
            summary_path = os.path.join(visualizer.visualization_dir, 'evaluation_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"\nEvaluation summary saved to: {summary_path}")
    
    else:
        logger.info("Please specify --stage <name> or --all to evaluate models")


if __name__ == "__main__":
    main()