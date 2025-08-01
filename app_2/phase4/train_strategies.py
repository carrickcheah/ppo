"""
Unified training script for all Phase 4 strategy environments
Tests different scheduling scenarios with PPO
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Type

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from phase4.environments import (
    SmallBalancedEnvironment,
    SmallRushEnvironment,
    SmallBottleneckEnvironment,
    SmallComplexEnvironment
)


class StrategyTrainer:
    """Trains PPO on different strategy environments."""
    
    def __init__(self, output_dir: str = "/home/azureuser/ppo/app_2/phase4/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Environment configurations
        self.strategies = {
            'small_balanced': {
                'env_class': SmallBalancedEnvironment,
                'timesteps': 500000,
                'description': 'Balanced workload with mixed deadlines'
            },
            'small_rush': {
                'env_class': SmallRushEnvironment,
                'timesteps': 750000,  # More time for harder scenario
                'description': 'High pressure with urgent deadlines'
            },
            'small_bottleneck': {
                'env_class': SmallBottleneckEnvironment,
                'timesteps': 750000,  # More time for resource constraints
                'description': 'Resource constrained with high job/machine ratio'
            },
            'small_complex': {
                'env_class': SmallComplexEnvironment,
                'timesteps': 1000000,  # Most time for complex constraints
                'description': 'Complex dependencies and multi-machine jobs'
            }
        }
        
        # PPO hyperparameters (based on best from toy stages)
        self.ppo_params = {
            'learning_rate': 3e-4,
            'n_steps': 512,
            'batch_size': 64,
            'n_epochs': 20,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.05,  # Some exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        }
    
    def train_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Train a single strategy environment."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        config = self.strategies[strategy_name]
        env_class = config['env_class']
        timesteps = config['timesteps']
        
        print(f"\n{'='*60}")
        print(f"Training {strategy_name.upper()}")
        print(f"Description: {config['description']}")
        print(f"Timesteps: {timesteps:,}")
        print(f"{'='*60}\n")
        
        # Create environment
        def make_env():
            env = env_class(verbose=False)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Create model
        model = PPO('MlpPolicy', env, verbose=1, **self.ppo_params)
        
        # Setup callbacks
        checkpoint_dir = os.path.join(self.output_dir, strategy_name, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=checkpoint_dir,
            name_prefix=f"{strategy_name}_checkpoint"
        )
        
        # Training
        start_time = time.time()
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=checkpoint_callback,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
        
        training_time = time.time() - start_time
        
        # Evaluation
        print(f"\nEvaluating {strategy_name}...")
        eval_results = self.evaluate_model(model, env_class, n_episodes=50)
        
        # Save model
        model_path = os.path.join(self.output_dir, strategy_name, f"{strategy_name}_final.zip")
        model.save(model_path)
        
        # Prepare results
        results = {
            'strategy': strategy_name,
            'training_time_min': training_time / 60,
            'timesteps': timesteps,
            'eval_results': eval_results,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, strategy_name, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_results(strategy_name, eval_results, training_time)
        
        return results
    
    def evaluate_model(self, model, env_class: Type, n_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate trained model."""
        env = env_class(verbose=False)
        
        results = {
            'episodes': n_episodes,
            'completion_rates': [],
            'rewards': [],
            'steps': [],
            'strategy_metrics': {}
        }
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = done or truncated
            
            # Get metrics
            completion_rate = len(env.scheduled_jobs) / env.total_tasks if env.total_tasks > 0 else 0
            results['completion_rates'].append(completion_rate)
            results['rewards'].append(ep_reward)
            results['steps'].append(ep_steps)
        
        # Calculate statistics
        results['avg_completion'] = np.mean(results['completion_rates'])
        results['std_completion'] = np.std(results['completion_rates'])
        results['min_completion'] = np.min(results['completion_rates'])
        results['max_completion'] = np.max(results['completion_rates'])
        results['success_rate'] = sum(1 for r in results['completion_rates'] if r >= 0.7) / n_episodes
        results['avg_reward'] = np.mean(results['rewards'])
        results['avg_steps'] = np.mean(results['steps'])
        
        # Get strategy-specific metrics
        if hasattr(env, 'get_metrics_summary'):
            results['strategy_metrics'] = env.get_metrics_summary()
        
        return results
    
    def _print_results(self, strategy_name: str, results: Dict, training_time: float):
        """Print formatted results."""
        print(f"\n{'='*50}")
        print(f"RESULTS: {strategy_name}")
        print(f"{'='*50}")
        print(f"Training time: {training_time/60:.1f} minutes")
        print(f"Average completion: {results['avg_completion']:.1%} (±{results['std_completion']:.1%})")
        print(f"Success rate (≥70%): {results['success_rate']:.1%}")
        print(f"Min/Max completion: {results['min_completion']:.1%} / {results['max_completion']:.1%}")
        print(f"Average reward: {results['avg_reward']:.1f}")
        print(f"Average steps: {results['avg_steps']:.1f}")
        
        if results.get('strategy_metrics'):
            print("\nStrategy-specific metrics:")
            for key, value in results['strategy_metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    def train_all_strategies(self):
        """Train all strategy environments."""
        print("\nPHASE 4: STRATEGY DEVELOPMENT")
        print("Training PPO on different scheduling scenarios")
        print("-" * 60)
        
        all_results = {}
        
        for strategy_name in self.strategies:
            try:
                results = self.train_strategy(strategy_name)
                all_results[strategy_name] = results
            except Exception as e:
                print(f"\nError training {strategy_name}: {e}")
                all_results[strategy_name] = {'error': str(e)}
        
        # Final summary
        self._print_final_summary(all_results)
        
        # Save combined results
        summary_path = os.path.join(self.output_dir, 'strategy_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def _print_final_summary(self, all_results: Dict):
        """Print final summary of all strategies."""
        print("\n" + "="*70)
        print("PHASE 4 SUMMARY: STRATEGY DEVELOPMENT")
        print("="*70)
        print(f"\n{'Strategy':<20} {'Completion':<15} {'Success Rate':<15} {'Status'}")
        print("-"*65)
        
        for strategy, results in all_results.items():
            if 'error' in results:
                print(f"{strategy:<20} {'ERROR':<15} {'ERROR':<15} Failed")
            else:
                eval_results = results['eval_results']
                completion = f"{eval_results['avg_completion']:.1%}"
                success = f"{eval_results['success_rate']:.1%}"
                status = "✓ Pass" if eval_results['success_rate'] >= 0.5 else "✗ Needs work"
                print(f"{strategy:<20} {completion:<15} {success:<15} {status}")
        
        # Overall assessment
        successful = sum(
            1 for r in all_results.values() 
            if 'eval_results' in r and r['eval_results']['success_rate'] >= 0.5
        )
        
        print(f"\nOverall: {successful}/{len(all_results)} strategies achieved ≥50% success rate")
        
        if successful >= 2:
            print("\n✓ Phase 4 objectives met! Ready for medium-scale environments.")
        else:
            print("\n✗ More tuning needed before scaling up.")


def main():
    """Main training function."""
    trainer = StrategyTrainer()
    
    # Option to train individual strategies or all
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO on strategy environments')
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['all', 'small_balanced', 'small_rush', 
                               'small_bottleneck', 'small_complex'],
                       help='Which strategy to train')
    
    args = parser.parse_args()
    
    if args.strategy == 'all':
        trainer.train_all_strategies()
    else:
        trainer.train_strategy(args.strategy)


if __name__ == "__main__":
    main()