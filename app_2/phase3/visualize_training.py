"""
Visualize Training Progress

Creates charts and reports for curriculum learning progress.
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualizes curriculum training progress."""
    
    def __init__(self, checkpoint_dir: str, tensorboard_dir: str):
        """Initialize visualizer."""
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.output_dir = 'phase3/visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_tensorboard_data(self, stage_name: str) -> pd.DataFrame:
        """Load data from TensorBoard logs."""
        log_dir = os.path.join(self.tensorboard_dir, stage_name)
        
        if not os.path.exists(log_dir):
            return pd.DataFrame()
            
        # Find event file
        event_files = [f for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
        if not event_files:
            return pd.DataFrame()
            
        event_path = os.path.join(log_dir, event_files[0])
        
        # Load events
        ea = EventAccumulator(event_path)
        ea.Reload()
        
        # Extract metrics
        data = {
            'step': [],
            'episode_reward': [],
            'loss': [],
            'learning_rate': []
        }
        
        # Episode rewards
        if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
            for event in ea.Scalars('rollout/ep_rew_mean'):
                data['step'].append(event.step)
                data['episode_reward'].append(event.value)
                
        # Loss
        if 'train/loss' in ea.Tags()['scalars']:
            for event in ea.Scalars('train/loss'):
                if event.step < len(data['step']):
                    idx = data['step'].index(event.step)
                    data['loss'].append(event.value)
                    
        return pd.DataFrame(data)
        
    def plot_stage_progress(self, stage_name: str):
        """Plot progress for a single stage."""
        # Load data
        df = self.load_tensorboard_data(stage_name)
        
        if df.empty:
            print(f"No data found for stage: {stage_name}")
            return
            
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Training Progress: {stage_name}', fontsize=16)
        
        # Episode rewards
        ax1 = axes[0]
        ax1.plot(df['step'], df['episode_reward'], alpha=0.6, label='Raw')
        
        # Rolling average
        if len(df) > 10:
            rolling_mean = df['episode_reward'].rolling(window=10).mean()
            ax1.plot(df['step'], rolling_mean, linewidth=2, label='Moving Average')
            
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        if 'loss' in df.columns and not df['loss'].empty:
            ax2 = axes[1]
            ax2.plot(df['step'][:len(df['loss'])], df['loss'], alpha=0.8, color='orange')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, f'{stage_name}_progress.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {output_path}")
        
    def plot_curriculum_overview(self, stages: List[str]):
        """Plot overview of entire curriculum."""
        # Collect data from all stages
        all_data = []
        
        for stage in stages:
            df = self.load_tensorboard_data(stage)
            if not df.empty:
                df['stage'] = stage
                all_data.append(df)
                
        if not all_data:
            print("No training data found")
            return
            
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each stage
        cumulative_steps = 0
        stage_boundaries = []
        
        for stage in stages:
            stage_data = combined_df[combined_df['stage'] == stage]
            if not stage_data.empty:
                # Adjust steps to be cumulative
                adjusted_steps = stage_data['step'] + cumulative_steps
                
                # Plot
                ax.plot(
                    adjusted_steps,
                    stage_data['episode_reward'],
                    alpha=0.6,
                    label=stage
                )
                
                # Mark stage boundary
                stage_boundaries.append(cumulative_steps)
                cumulative_steps = adjusted_steps.max()
                
        # Add stage boundaries
        for boundary in stage_boundaries[1:]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            
        ax.set_xlabel('Total Training Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Curriculum Learning Progress Across All Stages')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'curriculum_overview.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved curriculum overview: {output_path}")
        
    def generate_training_report(self, config_path: str):
        """Generate comprehensive training report."""
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Load training state
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
        else:
            state = {'completed_stages': [], 'current_stage_idx': 0}
            
        # Generate report
        report = []
        report.append("# Phase 3 Training Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Training Progress")
        report.append(f"- Completed Stages: {len(state['completed_stages'])}/16")
        report.append(f"- Current Stage: {state.get('current_stage_idx', 0)}")
        
        # Stage details
        report.append("\n## Stage Details")
        
        stages = [
            'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
            'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
            'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
            'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
        ]
        
        for stage in stages:
            stage_config = config.get(stage, {})
            status = "✓ Completed" if stage in state['completed_stages'] else "○ Pending"
            
            report.append(f"\n### {stage_config.get('name', stage)} {status}")
            report.append(f"- Jobs: {stage_config.get('jobs', 'N/A')}")
            report.append(f"- Machines: {stage_config.get('machines', 'N/A')}")
            report.append(f"- Timesteps: {stage_config.get('timesteps', 'N/A'):,}")
            report.append(f"- Description: {stage_config.get('description', 'N/A')}")
            
            # Check for best model
            best_model_path = os.path.join(self.checkpoint_dir, stage, 'best_model.zip')
            if os.path.exists(best_model_path):
                report.append(f"- Best Model: Available")
                
        # Training tips
        report.append("\n## Training Tips")
        report.append("- Monitor TensorBoard for real-time progress: `tensorboard --logdir phase3/tensorboard`")
        report.append("- Resume training from any stage: `python train_curriculum.py --start-stage <stage_name>`")
        report.append("- Train single stage: `python train_curriculum.py --single-stage <stage_name>`")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'training_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        print(f"Generated training report: {report_path}")
        
    def plot_stage_comparison(self, stages: List[str]):
        """Compare performance across stages."""
        # Collect final metrics
        stage_metrics = []
        
        for stage in stages:
            df = self.load_tensorboard_data(stage)
            if not df.empty and 'episode_reward' in df.columns:
                # Get last 10% of training
                n_samples = max(1, len(df) // 10)
                final_rewards = df['episode_reward'].tail(n_samples)
                
                stage_metrics.append({
                    'stage': stage,
                    'mean_reward': final_rewards.mean(),
                    'std_reward': final_rewards.std(),
                    'max_reward': final_rewards.max()
                })
                
        if not stage_metrics:
            print("No metrics to compare")
            return
            
        # Create DataFrame
        metrics_df = pd.DataFrame(stage_metrics)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(metrics_df))
        ax.bar(x, metrics_df['mean_reward'], yerr=metrics_df['std_reward'], 
               capsize=5, alpha=0.7, label='Mean ± Std')
        ax.scatter(x, metrics_df['max_reward'], color='red', s=50, 
                   zorder=5, label='Max Reward')
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['stage'], rotation=45, ha='right')
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Performance Comparison Across Stages')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.output_dir, 'stage_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved stage comparison: {output_path}")


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='phase3/checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default='phase3/tensorboard',
        help='TensorBoard log directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase3_curriculum_config.yaml',
        help='Training configuration'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default=None,
        help='Visualize specific stage'
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = TrainingVisualizer(args.checkpoint_dir, args.tensorboard_dir)
    
    # Define stages
    stages = [
        'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
        'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
        'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
        'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
    ]
    
    if args.stage:
        # Visualize single stage
        viz.plot_stage_progress(args.stage)
    else:
        # Visualize all
        viz.plot_curriculum_overview(stages)
        viz.plot_stage_comparison(stages)
        viz.generate_training_report(args.config)
        
        # Individual stage plots
        for stage in stages:
            viz.plot_stage_progress(stage)


if __name__ == '__main__':
    main()