"""
Evaluate Improved Foundation Models
Tests scheduling performance of the improved trained models
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal


class ImprovedModelEvaluator:
    """Evaluates improved foundation models."""
    
    def __init__(self):
        self.foundation_stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
        self.checkpoint_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation_improved"
        self.results_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_stage(self, stage_name: str, n_episodes: int = 10) -> Dict:
        """Evaluate a single stage model."""
        print(f"\n{'='*60}")
        print(f"Evaluating {stage_name.upper()}")
        print(f"{'='*60}")
        
        # Check if model exists
        model_path = os.path.join(self.checkpoint_dir, stage_name, "improved_model.zip")
        vec_norm_path = os.path.join(self.checkpoint_dir, stage_name, "vec_normalize.pkl")
        
        if not os.path.exists(model_path):
            print(f"No improved model found for {stage_name}")
            return {
                'stage': stage_name,
                'status': 'not_found',
                'scheduling_rate': 0.0
            }
        
        # Create environment
        env = CurriculumEnvironmentReal(stage_name=stage_name, verbose=False)
        
        # Load model
        model = PPO.load(model_path)
        
        # Evaluation metrics
        metrics = {
            'total_episodes': n_episodes,
            'total_scheduled': 0,
            'total_possible': 0,
            'rewards': [],
            'episode_lengths': [],
            'scheduling_rates': [],
            'completion_times': [],
            'late_jobs': 0,
            'on_time_jobs': 0,
            'machine_utilization': []
        }
        
        # Run evaluation episodes
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = done or truncated
            
            # Collect episode metrics
            scheduled = len(env.scheduled_jobs)
            total = env.total_tasks
            rate = scheduled / total if total > 0 else 0
            
            metrics['total_scheduled'] += scheduled
            metrics['total_possible'] += total
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(steps)
            metrics['scheduling_rates'].append(rate)
            
            # Analyze job outcomes
            for job_key, job_info in env.job_assignments.items():
                family_id = job_key.split('_seq')[0]
                if family_id in env.families:
                    lcd_days = env.families[family_id]['lcd_days_remaining']
                    completion_days = job_info['end'] / 24.0
                    
                    if completion_days <= lcd_days:
                        metrics['on_time_jobs'] += 1
                    else:
                        metrics['late_jobs'] += 1
            
            # Calculate machine utilization
            if env.current_time > 0:
                total_utilization = 0
                for machine_id in env.machine_ids:
                    busy_time = sum(job['end'] - job['start'] 
                                  for job in env.machine_schedules[machine_id])
                    utilization = busy_time / env.current_time
                    total_utilization += utilization
                
                avg_utilization = total_utilization / len(env.machine_ids)
                metrics['machine_utilization'].append(avg_utilization)
            
            print(f"Episode {episode + 1}: Scheduled {scheduled}/{total} "
                  f"({rate:.1%}), Reward: {episode_reward:.0f}")
        
        # Calculate summary statistics
        overall_rate = metrics['total_scheduled'] / metrics['total_possible'] if metrics['total_possible'] > 0 else 0
        
        summary = {
            'stage': stage_name,
            'status': 'evaluated',
            'scheduling_rate': overall_rate,
            'avg_reward': np.mean(metrics['rewards']),
            'std_reward': np.std(metrics['rewards']),
            'avg_episode_length': np.mean(metrics['episode_lengths']),
            'on_time_rate': metrics['on_time_jobs'] / (metrics['on_time_jobs'] + metrics['late_jobs']) 
                           if (metrics['on_time_jobs'] + metrics['late_jobs']) > 0 else 0,
            'avg_machine_utilization': np.mean(metrics['machine_utilization']) if metrics['machine_utilization'] else 0,
            'detailed_metrics': metrics
        }
        
        print(f"\nSummary for {stage_name}:")
        print(f"  - Overall scheduling rate: {overall_rate:.1%}")
        print(f"  - Average reward: {summary['avg_reward']:.0f}")
        print(f"  - On-time delivery rate: {summary['on_time_rate']:.1%}")
        print(f"  - Machine utilization: {summary['avg_machine_utilization']:.1%}")
        
        return summary
    
    def evaluate_all_stages(self):
        """Evaluate all foundation stages."""
        print("="*60)
        print("EVALUATING IMPROVED FOUNDATION MODELS")
        print("="*60)
        
        all_results = {}
        
        for stage in self.foundation_stages:
            results = self.evaluate_stage(stage)
            all_results[stage] = results
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.results_dir, f"q_improved_evaluation_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print("\nScheduling Rates:")
        
        for stage, results in all_results.items():
            if results['status'] == 'evaluated':
                print(f"  - {stage}: {results['scheduling_rate']:.1%}")
            else:
                print(f"  - {stage}: Not found")
        
        print(f"\nResults saved to: {results_path}")
        
        return all_results
    
    def compare_with_original(self):
        """Compare improved results with original training."""
        print("\n" + "="*60)
        print("IMPROVEMENT COMPARISON")
        print("="*60)
        
        # Original results (from earlier analysis)
        original_rates = {
            'toy_easy': 0.263,
            'toy_normal': 0.0,
            'toy_hard': 0.0,
            'toy_multi': 0.029
        }
        
        # Get improved results
        improved_results = self.evaluate_all_stages()
        
        print("\nComparison (Original → Improved):")
        for stage in self.foundation_stages:
            original = original_rates.get(stage, 0) * 100
            if stage in improved_results and improved_results[stage]['status'] == 'evaluated':
                improved = improved_results[stage]['scheduling_rate'] * 100
                change = improved - original
                print(f"  - {stage}: {original:.1f}% → {improved:.1f}% "
                      f"({'+'if change >= 0 else ''}{change:.1f}%)")
            else:
                print(f"  - {stage}: {original:.1f}% → Not available")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate improved foundation models")
    parser.add_argument('--compare', action='store_true', 
                       help='Compare with original results')
    parser.add_argument('--stage', type=str, default=None,
                       help='Evaluate specific stage only')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    evaluator = ImprovedModelEvaluator()
    
    if args.stage:
        # Evaluate single stage
        results = evaluator.evaluate_stage(args.stage, n_episodes=args.episodes)
    elif args.compare:
        # Compare with original
        evaluator.compare_with_original()
    else:
        # Evaluate all stages
        evaluator.evaluate_all_stages()


if __name__ == "__main__":
    main()