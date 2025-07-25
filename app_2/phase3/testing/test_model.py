"""
Test Trained Models

Evaluate performance of trained PPO models on various scenarios.
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.curriculum_env import CurriculumSchedulingEnv


class ModelTester:
    """Tests trained models on various scenarios."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize tester."""
        self.checkpoint_dir = checkpoint_dir
        self.results = []
        
    def load_model(self, stage_name: str) -> PPO:
        """Load trained model for stage."""
        model_path = os.path.join(self.checkpoint_dir, f'{stage_name}_final.zip')
        
        if not os.path.exists(model_path):
            # Try best model
            model_path = os.path.join(self.checkpoint_dir, stage_name, 'best_model.zip')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for stage: {stage_name}")
            
        return PPO.load(model_path)
        
    def test_scenario(
        self,
        model: PPO,
        env_config: Dict[str, Any],
        snapshot_path: str = None,
        n_episodes: int = 10
    ) -> Dict[str, Any]:
        """Test model on specific scenario."""
        # Create environment
        env = CurriculumSchedulingEnv(
            stage_config=env_config,
            snapshot_path=snapshot_path,
            seed=42
        )
        
        # Wrap for consistency
        env = DummyVecEnv([lambda: env])
        
        # Check for normalization
        vec_norm_path = os.path.join(
            self.checkpoint_dir,
            f"{env_config.get('stage_name', 'test')}_vec_normalize.pkl"
        )
        
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
            
        # Run episodes
        episode_results = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                if done:
                    # Extract metrics
                    if 'episode_metrics' in info[0]:
                        metrics = info[0]['episode_metrics']
                        metrics['total_reward'] = episode_reward
                        metrics['steps'] = step_count
                        episode_results.append(metrics)
                        
        # Aggregate results
        if episode_results:
            aggregated = {
                'n_episodes': len(episode_results),
                'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
                'avg_completion_rate': np.mean([r['jobs_completed'] / env_config['jobs'] for r in episode_results]),
                'avg_late_rate': np.mean([r['jobs_late'] / max(1, r['jobs_completed']) for r in episode_results]),
                'avg_utilization': np.mean([r['machine_utilization'] for r in episode_results]),
                'avg_makespan': np.mean([r['makespan'] for r in episode_results]),
                'episodes': episode_results
            }
        else:
            aggregated = {'error': 'No episodes completed'}
            
        env.close()
        return aggregated
        
    def test_model_progression(self, stages: List[str], test_configs: List[Dict]):
        """Test how models perform across different scenarios."""
        results = {}
        
        for stage in stages:
            print(f"\nTesting model: {stage}")
            
            try:
                model = self.load_model(stage)
                results[stage] = {}
                
                for test_config in test_configs:
                    test_name = test_config['name']
                    print(f"  Testing on: {test_name}")
                    
                    # Test
                    test_results = self.test_scenario(
                        model,
                        test_config['env_config'],
                        test_config.get('snapshot_path'),
                        n_episodes=5
                    )
                    
                    results[stage][test_name] = test_results
                    
                    # Print summary
                    if 'error' not in test_results:
                        print(f"    Avg Reward: {test_results['avg_reward']:.2f}")
                        print(f"    Completion: {test_results['avg_completion_rate']:.1%}")
                        print(f"    Late Rate: {test_results['avg_late_rate']:.1%}")
                        
            except Exception as e:
                print(f"  Error testing {stage}: {e}")
                results[stage] = {'error': str(e)}
                
        return results
        
    def generate_test_report(self, results: Dict[str, Any]):
        """Generate test report."""
        report = []
        report.append("# Model Testing Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary table
        report.append("\n## Performance Summary")
        report.append("\n| Model Stage | Toy (10 jobs) | Small (30 jobs) | Medium (150 jobs) | Production (295 jobs) |")
        report.append("|-------------|---------------|-----------------|-------------------|----------------------|")
        
        for stage, stage_results in results.items():
            if isinstance(stage_results, dict) and 'error' not in stage_results:
                row = [stage]
                
                for test_name in ['toy_test', 'small_test', 'medium_test', 'production_test']:
                    if test_name in stage_results:
                        test_data = stage_results[test_name]
                        if 'error' not in test_data:
                            score = f"{test_data['avg_completion_rate']:.0%}"
                        else:
                            score = "Error"
                    else:
                        score = "N/A"
                    row.append(score)
                    
                report.append(f"| {' | '.join(row)} |")
                
        # Detailed results
        report.append("\n## Detailed Results")
        
        for stage, stage_results in results.items():
            report.append(f"\n### {stage}")
            
            if isinstance(stage_results, dict) and 'error' not in stage_results:
                for test_name, test_data in stage_results.items():
                    report.append(f"\n#### {test_name}")
                    
                    if 'error' not in test_data:
                        report.append(f"- Episodes: {test_data['n_episodes']}")
                        report.append(f"- Avg Reward: {test_data['avg_reward']:.2f}")
                        report.append(f"- Completion Rate: {test_data['avg_completion_rate']:.1%}")
                        report.append(f"- Late Rate: {test_data['avg_late_rate']:.1%}")
                        report.append(f"- Machine Utilization: {test_data['avg_utilization']:.1%}")
                        report.append(f"- Avg Makespan: {test_data['avg_makespan']:.1f} hours")
                    else:
                        report.append(f"- Error: {test_data['error']}")
            else:
                report.append(f"- Error: {stage_results.get('error', 'Unknown error')}")
                
        # Save report
        output_dir = 'phase3/test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'test_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        # Save raw results
        results_path = os.path.join(output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nTest report saved to: {report_path}")
        print(f"Raw results saved to: {results_path}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='phase3/checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default=None,
        help='Test specific stage model'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all available models'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(args.checkpoint_dir)
    
    # Define test scenarios
    test_configs = [
        {
            'name': 'toy_test',
            'env_config': {
                'name': 'Toy Test',
                'jobs': 10,
                'machines': 5,
                'data_source': 'synthetic'
            }
        },
        {
            'name': 'small_test',
            'env_config': {
                'name': 'Small Test',
                'jobs': 30,
                'machines': 15,
                'data_source': 'synthetic'
            }
        },
        {
            'name': 'medium_test',
            'env_config': {
                'name': 'Medium Test',
                'jobs': 150,
                'machines': 40,
                'data_source': 'snapshot_normal_subset'
            },
            'snapshot_path': 'phase3/snapshots/snapshot_normal.json'
        },
        {
            'name': 'production_test',
            'env_config': {
                'name': 'Production Test',
                'jobs': 295,
                'machines': 145,
                'data_source': 'snapshot_normal'
            },
            'snapshot_path': 'phase3/snapshots/snapshot_normal.json'
        }
    ]
    
    # Determine which models to test
    if args.stage:
        stages = [args.stage]
    elif args.all:
        stages = [
            'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
            'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
            'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
            'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
        ]
    else:
        # Test latest available model
        stages = ['production_expert']  # Default to final stage
        
    # Run tests
    results = tester.test_model_progression(stages, test_configs)
    
    # Generate report
    tester.generate_test_report(results)


if __name__ == '__main__':
    main()