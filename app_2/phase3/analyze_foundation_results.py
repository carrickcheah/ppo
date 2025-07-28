"""
Analyze Foundation Training Results
Generates comprehensive logs, results, and charts for the 4 foundation stages
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FoundationResultsAnalyzer:
    """Analyzes and generates reports for foundation training results."""
    
    def __init__(self):
        self.log_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/logs"
        self.results_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/results"
        self.vis_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations"
        self.checkpoint_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Foundation stages
        self.stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
        self.stage_info = {
            'toy_easy': {'jobs': 5, 'machines': 3, 'focus': 'Sequence Learning'},
            'toy_normal': {'jobs': 10, 'machines': 5, 'focus': 'Deadline Learning'},
            'toy_hard': {'jobs': 15, 'machines': 5, 'focus': 'Priority Learning'},
            'toy_multi': {'jobs': 10, 'machines': 8, 'focus': 'Multi-Machine Learning'}
        }
        
        # Training results from the run
        self.training_results = {
            'toy_easy': {
                'initial_reward': 763,
                'final_reward': 1740,
                'scheduling_rate': 0.526,
                'training_time': 0.2,
                'episodes': 720
            },
            'toy_normal': {
                'initial_reward': -380,
                'final_reward': 145,
                'scheduling_rate': 0.440,
                'training_time': 0.2,
                'episodes': 179
            },
            'toy_hard': {
                'initial_reward': -4430,
                'final_reward': -4070,
                'scheduling_rate': 0.200,
                'training_time': 0.4,
                'episodes': 601
            },
            'toy_multi': {
                'initial_reward': -1530,
                'final_reward': 1280,
                'scheduling_rate': 0.575,
                'training_time': 0.3,
                'episodes': 241
            }
        }
    
    def generate_training_log(self):
        """Generate detailed training log."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_content = []
        
        log_content.append("="*80)
        log_content.append("FOUNDATION TRAINING LOG")
        log_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append("="*80)
        log_content.append("")
        
        # Summary
        log_content.append("TRAINING SUMMARY")
        log_content.append("-"*40)
        log_content.append(f"Total Stages Trained: {len(self.stages)}")
        log_content.append(f"Total Training Time: {sum(r['training_time'] for r in self.training_results.values()):.1f} minutes")
        log_content.append(f"Total Episodes: {sum(r['episodes'] for r in self.training_results.values())}")
        log_content.append("")
        
        # Stage-by-stage analysis
        for stage in self.stages:
            info = self.stage_info[stage]
            results = self.training_results[stage]
            
            log_content.append(f"\nSTAGE: {stage.upper()}")
            log_content.append("-"*40)
            log_content.append(f"Configuration:")
            log_content.append(f"  - Jobs: {info['jobs']}")
            log_content.append(f"  - Machines: {info['machines']}")
            log_content.append(f"  - Learning Focus: {info['focus']}")
            log_content.append(f"")
            log_content.append(f"Training Results:")
            log_content.append(f"  - Initial Reward: {results['initial_reward']:,.0f}")
            log_content.append(f"  - Final Reward: {results['final_reward']:,.0f}")
            log_content.append(f"  - Improvement: {((results['final_reward'] - results['initial_reward']) / abs(results['initial_reward']) * 100):.1f}%")
            log_content.append(f"  - Scheduling Rate: {results['scheduling_rate']:.1%}")
            log_content.append(f"  - Training Time: {results['training_time']} minutes")
            log_content.append(f"  - Episodes: {results['episodes']}")
            
            # Learning assessment
            log_content.append(f"")
            log_content.append(f"Learning Assessment:")
            if stage == 'toy_easy':
                if results['scheduling_rate'] > 0.5:
                    log_content.append(f"  ✓ Successfully learned sequence constraints")
                else:
                    log_content.append(f"  ⚠ Partial learning of sequence constraints")
            elif stage == 'toy_normal':
                if results['final_reward'] > 0:
                    log_content.append(f"  ✓ Learned to optimize for deadlines")
                else:
                    log_content.append(f"  ⚠ Still struggling with deadline optimization")
            elif stage == 'toy_hard':
                if results['scheduling_rate'] < 0.3:
                    log_content.append(f"  ⚠ Difficulty with increased complexity")
                    log_content.append(f"  - May need more training time or adjusted hyperparameters")
                else:
                    log_content.append(f"  ✓ Handling priority jobs adequately")
            elif stage == 'toy_multi':
                if results['scheduling_rate'] > 0.5:
                    log_content.append(f"  ✓ Successfully learned multi-machine coordination")
                else:
                    log_content.append(f"  ⚠ Needs improvement in multi-machine handling")
        
        # Overall conclusions
        log_content.append("\n" + "="*40)
        log_content.append("OVERALL CONCLUSIONS")
        log_content.append("="*40)
        log_content.append("")
        
        avg_scheduling_rate = np.mean([r['scheduling_rate'] for r in self.training_results.values()])
        log_content.append(f"Average Scheduling Rate: {avg_scheduling_rate:.1%}")
        
        if avg_scheduling_rate > 0.4:
            log_content.append("✓ Foundation training successful - model learned core concepts")
        else:
            log_content.append("⚠ Foundation training needs improvement")
        
        log_content.append("")
        log_content.append("Recommendations:")
        log_content.append("1. Toy_hard stage showed lowest performance (20%) - consider:")
        log_content.append("   - Increasing training timesteps for this stage")
        log_content.append("   - Adjusting reward structure for better priority learning")
        log_content.append("2. Overall positive trend in learning progression")
        log_content.append("3. Model ready for more complex stages with careful monitoring")
        
        # Save log
        log_path = os.path.join(self.log_dir, f"q_foundation_training_log_{timestamp}.txt")
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_content))
        
        logger.info(f"Training log saved to: {log_path}")
        return log_path
    
    def generate_results_summary(self):
        """Generate structured results summary."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'total_stages': len(self.stages),
                'framework': 'PPO with Stable Baselines3',
                'environment': 'CurriculumEnvironmentReal',
                'data_source': 'MariaDB Production Data'
            },
            'hyperparameters': {
                'learning_rate': 3e-4,
                'batch_size': 64,
                'n_epochs': 10,
                'ent_coef': 0.1,
                'timesteps_per_stage': 25000
            },
            'stage_results': {},
            'performance_metrics': {},
            'learning_outcomes': {}
        }
        
        # Add stage results
        for stage in self.stages:
            results['stage_results'][stage] = {
                **self.training_results[stage],
                **self.stage_info[stage],
                'reward_improvement': self.training_results[stage]['final_reward'] - self.training_results[stage]['initial_reward'],
                'reward_improvement_pct': ((self.training_results[stage]['final_reward'] - self.training_results[stage]['initial_reward']) / 
                                          abs(self.training_results[stage]['initial_reward']) * 100)
            }
        
        # Calculate performance metrics
        results['performance_metrics'] = {
            'average_scheduling_rate': np.mean([r['scheduling_rate'] for r in self.training_results.values()]),
            'best_performing_stage': max(self.training_results.items(), key=lambda x: x[1]['scheduling_rate'])[0],
            'worst_performing_stage': min(self.training_results.items(), key=lambda x: x[1]['scheduling_rate'])[0],
            'total_training_time_minutes': sum(r['training_time'] for r in self.training_results.values()),
            'total_episodes': sum(r['episodes'] for r in self.training_results.values())
        }
        
        # Learning outcomes assessment
        results['learning_outcomes'] = {
            'sequence_learning': 'Successful' if self.training_results['toy_easy']['scheduling_rate'] > 0.5 else 'Partial',
            'deadline_awareness': 'Successful' if self.training_results['toy_normal']['final_reward'] > 0 else 'In Progress',
            'priority_handling': 'Needs Improvement' if self.training_results['toy_hard']['scheduling_rate'] < 0.3 else 'Adequate',
            'multi_machine_coordination': 'Successful' if self.training_results['toy_multi']['scheduling_rate'] > 0.5 else 'Partial'
        }
        
        # Save results
        results_path = os.path.join(self.results_dir, f"q_foundation_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results summary saved to: {results_path}")
        return results_path
    
    def generate_visualization_charts(self):
        """Generate comprehensive visualization charts."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Learning Progression Chart
        self._create_learning_progression_chart(timestamp)
        
        # 2. Scheduling Rate Comparison
        self._create_scheduling_rate_chart(timestamp)
        
        # 3. Reward Evolution Chart
        self._create_reward_evolution_chart(timestamp)
        
        # 4. Performance Radar Chart
        self._create_performance_radar_chart(timestamp)
        
        # 5. Training Efficiency Chart
        self._create_training_efficiency_chart(timestamp)
    
    def _create_learning_progression_chart(self, timestamp: str):
        """Create learning progression across stages."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        stages = list(self.stages)
        x = np.arange(len(stages))
        
        # Subplot 1: Reward Progression
        initial_rewards = [self.training_results[s]['initial_reward'] for s in stages]
        final_rewards = [self.training_results[s]['final_reward'] for s in stages]
        
        width = 0.35
        bars1 = ax1.bar(x - width/2, initial_rewards, width, label='Initial Reward', color='lightcoral')
        bars2 = ax1.bar(x + width/2, final_rewards, width, label='Final Reward', color='darkgreen')
        
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Foundation Training: Reward Progression', fontsize=14, weight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in stages])
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Subplot 2: Scheduling Rate
        scheduling_rates = [self.training_results[s]['scheduling_rate'] for s in stages]
        colors = ['green' if r > 0.5 else 'orange' if r > 0.3 else 'red' for r in scheduling_rates]
        
        bars3 = ax2.bar(x, scheduling_rates, color=colors, edgecolor='black')
        ax2.set_ylabel('Scheduling Rate', fontsize=12)
        ax2.set_title('Foundation Training: Task Scheduling Success Rate', fontsize=14, weight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in stages])
        ax2.set_ylim(0, 1.0)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Target (50%)')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars3, scheduling_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Add stage info
        for i, stage in enumerate(stages):
            info = self.stage_info[stage]
            ax2.text(i, -0.1, f"{info['jobs']}J/{info['machines']}M", 
                    ha='center', va='top', fontsize=9, style='italic')
        
        plt.tight_layout()
        chart_path = os.path.join(self.vis_dir, f"q_learning_progression_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Learning progression chart saved to: {chart_path}")
    
    def _create_scheduling_rate_chart(self, timestamp: str):
        """Create detailed scheduling rate analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stages = list(self.stages)
        rates = [self.training_results[s]['scheduling_rate'] for s in stages]
        
        # Create bar chart with gradient colors
        colors = plt.cm.RdYlGn([(r + 0.1) / 1.1 for r in rates])  # Normalize to 0-1 range
        bars = ax.bar(stages, rates, color=colors, edgecolor='black', linewidth=2)
        
        # Add target line
        ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Target Rate (50%)')
        ax.axhline(y=0.8, color='green', linestyle=':', linewidth=2, label='Excellent (80%)')
        
        # Customize
        ax.set_ylabel('Scheduling Success Rate', fontsize=14, weight='bold')
        ax.set_title('Foundation Stages: Scheduling Performance Analysis', fontsize=16, weight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in stages], rotation=15)
        
        # Add value labels and performance indicators
        for bar, rate, stage in zip(bars, rates, stages):
            # Value label
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=12, weight='bold')
            
            # Performance indicator
            if rate >= 0.8:
                indicator = '★★★'  # 3 stars
            elif rate >= 0.5:
                indicator = '★★'     # 2 stars
            elif rate >= 0.3:
                indicator = '★'         # 1 star
            else:
                indicator = '⚠'         # warning
            
            ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                   indicator, ha='center', va='bottom', fontsize=16)
            
            # Add focus area
            focus = self.stage_info[stage]['focus']
            ax.text(bar.get_x() + bar.get_width()/2., -0.08,
                   focus, ha='center', va='top', fontsize=9, style='italic', rotation=15)
        
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        chart_path = os.path.join(self.vis_dir, f"q_scheduling_rate_analysis_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Scheduling rate chart saved to: {chart_path}")
    
    def _create_reward_evolution_chart(self, timestamp: str):
        """Create reward evolution visualization."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        stages = list(self.stages)
        x = np.arange(len(stages))
        
        # Data
        initial = [self.training_results[s]['initial_reward'] for s in stages]
        final = [self.training_results[s]['final_reward'] for s in stages]
        improvement = [f - i for i, f in zip(initial, final)]
        
        # Create grouped bar chart
        width = 0.25
        bars1 = ax.bar(x - width, initial, width, label='Initial', color='#ff7f0e', alpha=0.8)
        bars2 = ax.bar(x, final, width, label='Final', color='#2ca02c', alpha=0.8)
        bars3 = ax.bar(x + width, improvement, width, label='Improvement', color='#1f77b4', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Training Stage', fontsize=12)
        ax.set_ylabel('Reward Value', fontsize=12)
        ax.set_title('Foundation Training: Reward Evolution by Stage', fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in stages])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add improvement percentage labels
        for i, (stage, imp) in enumerate(zip(stages, improvement)):
            pct = (imp / abs(initial[i])) * 100 if initial[i] != 0 else 0
            color = 'green' if pct > 0 else 'red'
            ax.text(i, ax.get_ylim()[1] * 0.9, f'{pct:+.0f}%', 
                   ha='center', va='bottom', color=color, weight='bold', fontsize=11)
        
        plt.tight_layout()
        chart_path = os.path.join(self.vis_dir, f"q_reward_evolution_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Reward evolution chart saved to: {chart_path}")
    
    def _create_performance_radar_chart(self, timestamp: str):
        """Create radar chart showing multi-dimensional performance."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Metrics for each stage
        categories = ['Scheduling\nRate', 'Reward\nGrowth', 'Training\nSpeed', 'Complexity\nHandling']
        num_vars = len(categories)
        
        # Compute angles
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Plot data for each stage
        for stage in self.stages:
            values = []
            
            # Scheduling Rate (0-1)
            values.append(self.training_results[stage]['scheduling_rate'])
            
            # Reward Growth (normalized to 0-1)
            initial = self.training_results[stage]['initial_reward']
            final = self.training_results[stage]['final_reward']
            growth = (final - initial) / (abs(initial) + 1000)  # Normalize
            values.append(min(1.0, max(0.0, growth + 0.5)))  # Scale to 0-1
            
            # Training Speed (inversely proportional to episodes)
            speed = 1.0 - (self.training_results[stage]['episodes'] / 1000.0)
            values.append(max(0.0, speed))
            
            # Complexity Handling (based on jobs/machines ratio and performance)
            complexity = self.stage_info[stage]['jobs'] / self.stage_info[stage]['machines']
            handling = self.training_results[stage]['scheduling_rate'] / (complexity / 3.0)
            values.append(min(1.0, handling))
            
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=stage.replace('_', ' ').title())
            ax.fill(angles, values, alpha=0.15)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['20%', '40%', '60%', '80%'], size=10)
        ax.set_title('Foundation Training: Multi-Dimensional Performance Analysis', 
                    size=14, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        chart_path = os.path.join(self.vis_dir, f"q_performance_radar_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance radar chart saved to: {chart_path}")
    
    def _create_training_efficiency_chart(self, timestamp: str):
        """Create training efficiency analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stages = list(self.stages)
        
        # Chart 1: Episodes vs Performance
        episodes = [self.training_results[s]['episodes'] for s in stages]
        rates = [self.training_results[s]['scheduling_rate'] for s in stages]
        
        scatter = ax1.scatter(episodes, rates, s=200, c=range(len(stages)), 
                            cmap='viridis', edgecolor='black', linewidth=2)
        
        # Add stage labels
        for i, stage in enumerate(stages):
            ax1.annotate(stage.replace('_', ' ').title(), 
                        (episodes[i], rates[i]), 
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=10, weight='bold')
        
        ax1.set_xlabel('Training Episodes', fontsize=12)
        ax1.set_ylabel('Scheduling Rate', fontsize=12)
        ax1.set_title('Training Efficiency: Episodes vs Performance', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.7)
        
        # Chart 2: Time Efficiency
        times = [self.training_results[s]['training_time'] for s in stages]
        jobs = [self.stage_info[s]['jobs'] for s in stages]
        efficiency = [r / t for r, t in zip(rates, times)]  # Performance per minute
        
        bars = ax2.bar(stages, efficiency, color=plt.cm.plasma(np.linspace(0, 1, len(stages))))
        
        ax2.set_ylabel('Performance per Minute', fontsize=12)
        ax2.set_title('Training Efficiency: Performance Gain Rate', fontsize=14, weight='bold')
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in stages], rotation=15)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add efficiency values
        for bar, eff in zip(bars, efficiency):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{eff:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        chart_path = os.path.join(self.vis_dir, f"q_training_efficiency_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training efficiency chart saved to: {chart_path}")
    
    def generate_all_outputs(self):
        """Generate all logs, results, and charts."""
        logger.info("="*60)
        logger.info("GENERATING FOUNDATION TRAINING ANALYSIS")
        logger.info("="*60)
        
        # 1. Generate training log
        log_path = self.generate_training_log()
        logger.info(f"✓ Training log generated: {log_path}")
        
        # 2. Generate results summary
        results_path = self.generate_results_summary()
        logger.info(f"✓ Results summary generated: {results_path}")
        
        # 3. Generate visualization charts
        self.generate_visualization_charts()
        logger.info(f"✓ All visualization charts generated")
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"\nOutputs saved to:")
        logger.info(f"  - Logs: {self.log_dir}")
        logger.info(f"  - Results: {self.results_dir}")
        logger.info(f"  - Visualizations: {self.vis_dir}")


def main():
    """Main entry point."""
    analyzer = FoundationResultsAnalyzer()
    analyzer.generate_all_outputs()


if __name__ == "__main__":
    main()