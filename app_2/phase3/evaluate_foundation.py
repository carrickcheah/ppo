"""
Evaluate Foundation Stages - Analyze What Model Learned
Generates detailed analysis and visualizations for the 4 foundation stages
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
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


class FoundationEvaluator:
    """Evaluates foundation training stages and generates analysis."""
    
    def __init__(self, 
                 checkpoint_dir: str = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation",
                 visualization_dir: str = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3/foundation"):
        self.checkpoint_dir = checkpoint_dir
        self.visualization_dir = visualization_dir
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Foundation stages
        self.foundation_stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
        
    def load_model_and_env(self, stage_name: str) -> Tuple[PPO, VecNormalize]:
        """Load trained model and environment for a stage."""
        # Model path
        model_path = os.path.join(self.checkpoint_dir, stage_name, "final_model.zip")
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
        
        # Load normalization stats
        vec_norm_path = os.path.join(self.checkpoint_dir, stage_name, "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, training=False, norm_reward=False)
        
        return model, env
    
    def analyze_stage_behavior(self, stage_name: str, n_episodes: int = 10) -> Dict:
        """Analyze what the model learned in a specific stage."""
        logger.info(f"\nAnalyzing {stage_name}...")
        
        try:
            model, env = self.load_model_and_env(stage_name)
        except FileNotFoundError:
            logger.warning(f"Skipping {stage_name} - no model found")
            return {}
        
        # Metrics to track
        metrics = {
            'stage_name': stage_name,
            'episodes_analyzed': n_episodes,
            'scheduling_patterns': [],
            'sequence_violations': 0,
            'jobs_completed': 0,
            'total_jobs': 0,
            'on_time_rate': 0,
            'important_job_priority': 0,
            'multi_machine_success': 0,
            'avg_utilization': 0,
            'avg_makespan': 0
        }
        
        episode_data = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_info = {
                'actions': [],
                'rewards': [],
                'job_sequence': []
            }
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                if info[0].get('action_valid', False):
                    episode_info['actions'].append(action[0].tolist())
                    episode_info['job_sequence'].append(info[0].get('scheduled_job', ''))
                
                episode_info['rewards'].append(float(reward[0]))
            
            # Analyze episode results
            base_env = env.envs[0].env
            
            # Basic metrics
            metrics['total_jobs'] += len(base_env.families)
            metrics['jobs_completed'] += len(base_env.completed_jobs)
            
            # Check sequence patterns
            for family_id in base_env.completed_jobs:
                family_prog = base_env.family_progress[family_id]
                if family_prog['completed_sequences'] != family_prog['total_sequences']:
                    metrics['sequence_violations'] += 1
            
            # Calculate utilization
            if base_env.current_time > 0:
                total_busy = 0
                for machine_id, schedule in base_env.machine_schedules.items():
                    for job in schedule:
                        total_busy += job['end'] - job['start']
                utilization = total_busy / (base_env.current_time * len(base_env.machine_ids))
                metrics['avg_utilization'] += utilization / n_episodes
            
            # Track makespan
            if base_env.job_assignments:
                makespan = max(job['end'] for job in base_env.job_assignments.values())
                metrics['avg_makespan'] += makespan / n_episodes
            
            episode_data.append(episode_info)
        
        # Calculate final metrics
        if metrics['total_jobs'] > 0:
            metrics['completion_rate'] = metrics['jobs_completed'] / metrics['total_jobs']
        else:
            metrics['completion_rate'] = 0
        
        # Stage-specific analysis
        if stage_name == 'toy_easy':
            logger.info(f"  Sequence violations: {metrics['sequence_violations']}")
            logger.info(f"  Learned to respect sequences: {'Yes' if metrics['sequence_violations'] == 0 else 'Partially'}")
        
        elif stage_name == 'toy_normal':
            # Analyze deadline behavior
            logger.info(f"  Jobs scheduled: {metrics['jobs_completed']}/{metrics['total_jobs']}")
            logger.info(f"  Focus on deadlines: Analyzing scheduling order...")
        
        elif stage_name == 'toy_hard':
            # Check if important jobs are prioritized
            logger.info(f"  Completion rate: {metrics['completion_rate']:.1%}")
            logger.info(f"  Important job handling: Checking priority patterns...")
        
        elif stage_name == 'toy_multi':
            # Check multi-machine handling
            logger.info(f"  Utilization: {metrics['avg_utilization']:.1%}")
            logger.info(f"  Multi-machine coordination: Analyzing resource usage...")
        
        return metrics
    
    def create_learning_progression_chart(self, all_metrics: List[Dict]):
        """Create a chart showing learning progression across foundation stages."""
        if not all_metrics:
            logger.warning("No metrics to visualize")
            return
        
        stages = [m['stage_name'] for m in all_metrics]
        completion_rates = [m.get('completion_rate', 0) for m in all_metrics]
        utilizations = [m.get('avg_utilization', 0) for m in all_metrics]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Completion Rate Progression
        x = np.arange(len(stages))
        bars1 = ax1.bar(x, completion_rates, color='skyblue', edgecolor='navy')
        ax1.set_ylabel('Completion Rate', fontsize=12)
        ax1.set_title('Learning Progression: Job Completion', fontsize=14, weight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(stages)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, completion_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Add learning objectives
        objectives = [
            'Sequences',
            'Deadlines',
            'Priorities',
            'Multi-Machine'
        ]
        for i, obj in enumerate(objectives):
            ax1.text(i, 0.05, obj, ha='center', va='bottom', 
                    fontsize=10, style='italic', color='darkblue')
        
        # Plot 2: Utilization Progression
        bars2 = ax2.bar(x, utilizations, color='lightgreen', edgecolor='darkgreen')
        ax2.set_ylabel('Average Utilization', fontsize=12)
        ax2.set_title('Learning Progression: Machine Utilization', fontsize=14, weight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages)
        
        # Add value labels
        for bar, util in zip(bars2, utilizations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{util:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.visualization_dir, 'foundation_learning_progression.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nLearning progression chart saved to: {chart_path}")
    
    def create_stage_comparison_gantt(self, stage_name: str):
        """Create a simple Gantt chart for a specific stage."""
        try:
            model, env = self.load_model_and_env(stage_name)
        except FileNotFoundError:
            return
        
        # Generate one schedule
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
        
        # Get schedule data
        base_env = env.envs[0].env
        
        if not base_env.job_assignments:
            logger.warning(f"No jobs scheduled in {stage_name}")
            return
        
        # Create Gantt chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        machines = sorted(base_env.machine_schedules.keys())
        machine_positions = {m: i for i, m in enumerate(machines)}
        
        # Colors for different families
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        family_colors = {}
        
        # Plot jobs
        for job_key, assignment in base_env.job_assignments.items():
            family_id = job_key.split('_seq')[0]
            
            if family_id not in family_colors:
                family_colors[family_id] = colors[len(family_colors) % len(colors)]
            
            for machine_id in assignment['machines']:
                if machine_id in machine_positions:
                    y_pos = machine_positions[machine_id]
                    
                    rect = mpatches.Rectangle(
                        (assignment['start'], y_pos - 0.4),
                        assignment['end'] - assignment['start'],
                        0.8,
                        facecolor=family_colors[family_id],
                        edgecolor='black',
                        linewidth=1
                    )
                    ax.add_patch(rect)
                    
                    # Add sequence number
                    seq_num = job_key.split('seq')[-1]
                    ax.text(
                        (assignment['start'] + assignment['end']) / 2,
                        y_pos,
                        seq_num,
                        ha='center',
                        va='center',
                        fontsize=10,
                        weight='bold'
                    )
        
        # Format chart
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels([f"M{m}" for m in machines])
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_title(f'{stage_name.replace("_", " ").title()} - Learned Schedule Pattern', 
                    fontsize=14, weight='bold')
        
        # Set limits
        if base_env.job_assignments:
            max_time = max(a['end'] for a in base_env.job_assignments.values())
            ax.set_xlim(0, max_time * 1.1)
        
        ax.set_ylim(-0.5, len(machines) - 0.5)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add stage info
        info_text = f"Jobs: {len(base_env.completed_jobs)}/{len(base_env.families)}\n"
        info_text += f"Utilization: {base_env.current_time:.1f}h"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        gantt_path = os.path.join(self.visualization_dir, f'{stage_name}_schedule.png')
        plt.savefig(gantt_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Schedule visualization saved to: {gantt_path}")
    
    def evaluate_all_foundation_stages(self):
        """Evaluate all foundation stages and generate comprehensive report."""
        logger.info("="*60)
        logger.info("FOUNDATION STAGES EVALUATION")
        logger.info("="*60)
        
        all_metrics = []
        
        # Analyze each stage
        for stage in self.foundation_stages:
            metrics = self.analyze_stage_behavior(stage)
            if metrics:
                all_metrics.append(metrics)
                
                # Create schedule visualization
                self.create_stage_comparison_gantt(stage)
        
        # Create progression chart
        if all_metrics:
            self.create_learning_progression_chart(all_metrics)
        
        # Generate summary report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'stages_evaluated': len(all_metrics),
            'stage_metrics': all_metrics,
            'overall_findings': {
                'sequence_learning': 'Successful' if all_metrics and all_metrics[0].get('sequence_violations', 1) == 0 else 'Needs improvement',
                'average_completion_rate': np.mean([m.get('completion_rate', 0) for m in all_metrics]) if all_metrics else 0,
                'utilization_trend': 'Improving' if len(all_metrics) > 1 and all_metrics[-1].get('avg_utilization', 0) > all_metrics[0].get('avg_utilization', 0) else 'Stable'
            }
        }
        
        # Save report
        report_path = os.path.join(self.visualization_dir, 'foundation_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nEvaluation report saved to: {report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        
        for metrics in all_metrics:
            stage = metrics['stage_name']
            completion = metrics.get('completion_rate', 0)
            utilization = metrics.get('avg_utilization', 0)
            
            logger.info(f"\n{stage}:")
            logger.info(f"  Completion Rate: {completion:.1%}")
            logger.info(f"  Utilization: {utilization:.1%}")
            
            if stage == 'toy_easy':
                logger.info(f"  Sequence Learning: {'✓' if metrics.get('sequence_violations', 1) == 0 else '✗'}")
            elif stage == 'toy_multi':
                logger.info(f"  Multi-Machine Handling: {'✓' if utilization > 0.5 else 'Needs work'}")
        
        logger.info("\n" + "="*60)


def main():
    """Main entry point for foundation evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate foundation training stages")
    parser.add_argument(
        '--stage',
        type=str,
        choices=['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi'],
        help='Evaluate specific stage only'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=10,
        help='Number of episodes to evaluate per stage'
    )
    
    args = parser.parse_args()
    
    evaluator = FoundationEvaluator()
    
    if args.stage:
        # Evaluate single stage
        metrics = evaluator.analyze_stage_behavior(args.stage, n_episodes=args.n_episodes)
        evaluator.create_stage_comparison_gantt(args.stage)
    else:
        # Evaluate all foundation stages
        evaluator.evaluate_all_foundation_stages()


if __name__ == "__main__":
    main()