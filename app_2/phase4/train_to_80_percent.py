"""
Train all Phase 4 strategies to 80% completion rate
Implements iterative training with performance monitoring
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from phase4.environments import (
    SmallBalancedEnvironment,
    SmallRushEnvironment,
    SmallBottleneckEnvironment,
    SmallComplexEnvironment
)


class StrategyTrainerTo80:
    """Trains each strategy until 80% completion rate is achieved."""
    
    def __init__(self, output_dir: str = "/home/azureuser/ppo/app_2/phase4/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Environment configurations
        self.strategies = {
            'small_balanced': {
                'env_class': SmallBalancedEnvironment,
                'target_completion': 0.80,
                'max_iterations': 10,
                'timesteps_per_iteration': 200000,
                'description': 'Balanced workload - easiest to achieve 80%'
            },
            'small_rush': {
                'env_class': SmallRushEnvironment,
                'target_completion': 0.80,
                'max_iterations': 15,
                'timesteps_per_iteration': 300000,
                'description': 'Rush orders - harder due to tight deadlines'
            },
            'small_bottleneck': {
                'env_class': SmallBottleneckEnvironment,
                'target_completion': 0.80,
                'max_iterations': 12,
                'timesteps_per_iteration': 250000,
                'description': 'Resource bottleneck - needs efficient allocation'
            },
            'small_complex': {
                'env_class': SmallComplexEnvironment,
                'target_completion': 0.80,
                'max_iterations': 20,
                'timesteps_per_iteration': 400000,
                'description': 'Complex multi-machine - hardest scenario'
            }
        }
        
        # Progressive PPO hyperparameters
        self.hyperparameter_schedules = [
            # Initial exploration
            {
                'learning_rate': 5e-4,
                'ent_coef': 0.1,
                'clip_range': 0.3,
                'n_epochs': 10,
                'batch_size': 64
            },
            # Balanced learning
            {
                'learning_rate': 3e-4,
                'ent_coef': 0.05,
                'clip_range': 0.2,
                'n_epochs': 15,
                'batch_size': 128
            },
            # Fine-tuning
            {
                'learning_rate': 1e-4,
                'ent_coef': 0.02,
                'clip_range': 0.15,
                'n_epochs': 20,
                'batch_size': 256
            }
        ]
    
    def evaluate_model(self, model, env_class, n_episodes: int = 10) -> Tuple[float, Dict]:
        """Evaluate model performance."""
        env = env_class(verbose=False)
        
        results = {
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': []
        }
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Extract metrics
            completion_rate = info.get('completion_rate', 0)
            results['completion_rates'].append(completion_rate)
            results['rewards'].append(episode_reward)
            
            on_time_rate = info.get('on_time_jobs', 0) / max(info.get('scheduled_jobs', 1), 1)
            results['on_time_rates'].append(on_time_rate)
            
            utilization = info.get('machine_utilization', 0)
            results['utilizations'].append(utilization)
        
        env.close()
        
        avg_completion = np.mean(results['completion_rates'])
        return avg_completion, results
    
    def train_strategy_to_target(self, strategy_name: str) -> Dict[str, Any]:
        """Train a single strategy until target completion rate is achieved."""
        config = self.strategies[strategy_name]
        env_class = config['env_class']
        target = config['target_completion']
        max_iterations = config['max_iterations']
        timesteps_per_iter = config['timesteps_per_iteration']
        
        print(f"\n{'='*70}")
        print(f"Training {strategy_name.upper()} to {target*100:.0f}% completion")
        print(f"Description: {config['description']}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}\n")
        
        # Create environment
        def make_env():
            env = env_class(verbose=False)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Training history
        history = {
            'iterations': [],
            'completion_rates': [],
            'rewards': [],
            'best_completion': 0,
            'best_model_path': None
        }
        
        model = None
        best_completion = 0
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Select hyperparameters based on progress
            hp_index = min(iteration // 3, len(self.hyperparameter_schedules) - 1)
            hyperparams = self.hyperparameter_schedules[hp_index]
            
            print(f"Using hyperparameter set {hp_index + 1}: LR={hyperparams['learning_rate']}, "
                  f"Entropy={hyperparams['ent_coef']}")
            
            # Create or update model
            if model is None:
                model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=hyperparams['learning_rate'],
                    n_steps=512,
                    batch_size=hyperparams['batch_size'],
                    n_epochs=hyperparams['n_epochs'],
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=hyperparams['clip_range'],
                    ent_coef=hyperparams['ent_coef'],
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[256, 256], vf=[256, 256])
                    ),
                    verbose=1
                )
            else:
                # Update learning rate and entropy
                model.learning_rate = hyperparams['learning_rate']
                model.ent_coef = hyperparams['ent_coef']
            
            # Train
            print(f"Training for {timesteps_per_iter:,} timesteps...")
            model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
            
            # Evaluate
            print("Evaluating...")
            avg_completion, results = self.evaluate_model(model, env_class, n_episodes=20)
            avg_reward = np.mean(results['rewards'])
            avg_on_time = np.mean(results['on_time_rates'])
            avg_utilization = np.mean(results['utilizations'])
            
            print(f"Completion Rate: {avg_completion*100:.1f}%")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"On-time Rate: {avg_on_time*100:.1f}%")
            print(f"Utilization: {avg_utilization*100:.1f}%")
            
            # Save if best
            if avg_completion > best_completion:
                best_completion = avg_completion
                model_path = os.path.join(
                    self.output_dir, 
                    f"{strategy_name}_best_model_{avg_completion*100:.0f}pct.zip"
                )
                model.save(model_path)
                history['best_model_path'] = model_path
                history['best_completion'] = best_completion
                print(f"New best model saved! ({best_completion*100:.1f}%)")
            
            # Record history
            history['iterations'].append(iteration + 1)
            history['completion_rates'].append(avg_completion)
            history['rewards'].append(avg_reward)
            
            # Check if target reached
            if avg_completion >= target:
                print(f"\n✓ TARGET REACHED! {avg_completion*100:.1f}% >= {target*100:.0f}%")
                break
            
            # Adjust strategy if stuck
            if iteration > 2 and all(
                abs(history['completion_rates'][-i] - history['completion_rates'][-i-1]) < 0.02 
                for i in range(1, min(3, len(history['completion_rates'])))
            ):
                print("Performance plateaued, increasing exploration...")
                model.ent_coef = min(0.2, model.ent_coef * 1.5)
        
        env.close()
        
        # Save final results
        results_path = os.path.join(self.output_dir, f"{strategy_name}_training_history.json")
        with open(results_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return {
            'strategy': strategy_name,
            'final_completion': best_completion,
            'target_reached': best_completion >= target,
            'iterations_used': len(history['iterations']),
            'history': history
        }
    
    def train_all_strategies(self):
        """Train all strategies to 80% target."""
        print(f"\nPhase 4 Training to 80% Completion Rate")
        print(f"Started at: {datetime.now()}")
        print("="*70)
        
        results = {}
        
        # Train in order of expected difficulty
        strategy_order = ['small_balanced', 'small_bottleneck', 'small_rush', 'small_complex']
        
        for strategy in strategy_order:
            start_time = time.time()
            result = self.train_strategy_to_target(strategy)
            training_time = time.time() - start_time
            
            result['training_time_minutes'] = training_time / 60
            results[strategy] = result
            
            print(f"\n{strategy} Summary:")
            print(f"  Final Completion: {result['final_completion']*100:.1f}%")
            print(f"  Target Reached: {'YES' if result['target_reached'] else 'NO'}")
            print(f"  Training Time: {result['training_time_minutes']:.1f} minutes")
            print(f"  Iterations: {result['iterations_used']}")
        
        # Save overall summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'strategies_at_80': sum(1 for r in results.values() if r['target_reached']),
            'average_completion': np.mean([r['final_completion'] for r in results.values()])
        }
        
        summary_path = os.path.join(self.output_dir, "phase4_training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*70)
        print("PHASE 4 TRAINING COMPLETE")
        print(f"Strategies at 80%: {summary['strategies_at_80']}/4")
        print(f"Average Completion: {summary['average_completion']*100:.1f}%")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)
        
        return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default=None,
                       help='Train specific strategy (default: all)')
    args = parser.parse_args()
    
    trainer = StrategyTrainerTo80()
    
    if args.strategy:
        result = trainer.train_strategy_to_target(args.strategy)
        print(f"\nFinal result for {args.strategy}:")
        print(f"Completion rate: {result['final_completion']*100:.1f}%")
        print(f"Target reached: {result['target_reached']}")
    else:
        trainer.train_all_strategies()