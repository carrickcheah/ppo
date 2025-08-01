"""
Train Small Complex Environment to 80% completion rate
Complex multi-machine - hardest scenario
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from environments.small_complex_env import SmallComplexEnvironment


class ComplexStrategyTrainer:
    """Trains Small Complex strategy until 80% completion rate is achieved."""
    
    def __init__(self, output_dir: str = "/home/azureuser/ppo/app_2/phase4/results/complex"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Strategy configuration
        self.config = {
            'env_class': SmallComplexEnvironment,
            'target_completion': 0.80,
            'max_iterations': 20,
            'timesteps_per_iteration': 400000,
            'description': 'Complex multi-machine - hardest scenario'
        }
        
        # Progressive PPO hyperparameters for complex strategy
        # Complex needs extensive exploration and longer training
        self.hyperparameter_schedules = [
            # Extensive exploration (iterations 1-6)
            {
                'learning_rate': 8e-4,
                'ent_coef': 0.2,  # High entropy for complex dependencies
                'clip_range': 0.35,
                'n_epochs': 15,
                'batch_size': 128,
                'phase': 'extensive-exploration'
            },
            # Dependency learning (iterations 7-12)
            {
                'learning_rate': 5e-4,
                'ent_coef': 0.1,
                'clip_range': 0.25,
                'n_epochs': 20,
                'batch_size': 256,
                'phase': 'dependency-learning'
            },
            # Multi-machine coordination (iterations 13-16)
            {
                'learning_rate': 3e-4,
                'ent_coef': 0.05,
                'clip_range': 0.2,
                'n_epochs': 25,
                'batch_size': 384,
                'phase': 'coordination-optimization'
            },
            # Final refinement (iterations 17+)
            {
                'learning_rate': 1e-4,
                'ent_coef': 0.02,
                'clip_range': 0.15,
                'n_epochs': 30,
                'batch_size': 512,
                'phase': 'final-refinement'
            }
        ]
    
    def evaluate_model(self, model, n_episodes: int = 10) -> tuple[float, dict]:
        """Evaluate model performance on complex environment."""
        env = self.config['env_class'](verbose=False)
        
        results = {
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': [],
            'late_penalties': [],
            'multi_machine_jobs_completed': [],
            'dependency_violations': [],
            'resource_conflicts': [],
            'avg_job_complexity': []
        }
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            # Track complex-specific metrics
            multi_machine_completed = 0
            dependency_violations = 0
            resource_conflicts = 0
            job_complexities = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Collect step metrics if available
                if 'multi_machine_job_completed' in info:
                    multi_machine_completed += info['multi_machine_job_completed']
                if 'dependency_violation' in info:
                    dependency_violations += info['dependency_violation']
                if 'resource_conflict' in info:
                    resource_conflicts += info['resource_conflict']
                if 'job_complexity' in info:
                    job_complexities.append(info['job_complexity'])
            
            # Extract metrics
            completion_rate = info.get('completion_rate', 0)
            results['completion_rates'].append(completion_rate)
            results['rewards'].append(episode_reward)
            
            on_time_rate = info.get('on_time_jobs', 0) / max(info.get('scheduled_jobs', 1), 1)
            results['on_time_rates'].append(on_time_rate)
            
            utilization = info.get('machine_utilization', 0)
            results['utilizations'].append(utilization)
            
            late_penalty = info.get('total_late_penalty', 0)
            results['late_penalties'].append(late_penalty)
            
            # Complex-specific metrics
            results['multi_machine_jobs_completed'].append(multi_machine_completed)
            results['dependency_violations'].append(dependency_violations)
            results['resource_conflicts'].append(resource_conflicts)
            results['avg_job_complexity'].append(np.mean(job_complexities) if job_complexities else 0)
        
        env.close()
        
        avg_completion = np.mean(results['completion_rates'])
        return avg_completion, results
    
    def train_to_target(self) -> dict:
        """Train complex strategy until target completion rate is achieved."""
        target = self.config['target_completion']
        max_iterations = self.config['max_iterations']
        timesteps_per_iter = self.config['timesteps_per_iteration']
        
        print(f"\n{'='*70}")
        print(f"Training SMALL COMPLEX STRATEGY to {target*100:.0f}% completion")
        print(f"Description: {self.config['description']}")
        print(f"Max iterations: {max_iterations}")
        print(f"Timesteps per iteration: {timesteps_per_iter:,}")
        print(f"{'='*70}\n")
        
        # Create environment
        def make_env():
            env = self.config['env_class'](verbose=False)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Training history
        history = {
            'strategy': 'small_complex',
            'iterations': [],
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': [],
            'late_penalties': [],
            'multi_machine_success_rate': [],
            'dependency_violation_rate': [],
            'resource_conflict_rate': [],
            'learning_rates': [],
            'entropy_coeffs': [],
            'best_completion': 0,
            'best_model_path': None,
            'training_phases': []
        }
        
        model = None
        best_completion = 0
        total_timesteps = 0
        
        # Track complex-specific improvements
        multi_machine_improvement = 0
        dependency_handling_improvement = 0
        
        for iteration in range(max_iterations):
            start_time = time.time()
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Select hyperparameters based on progress
            if iteration < 6:
                hp_index = 0
            elif iteration < 12:
                hp_index = 1
            elif iteration < 16:
                hp_index = 2
            else:
                hp_index = 3
            
            hyperparams = self.hyperparameter_schedules[hp_index]
            
            # Adaptive adjustments for complex scenarios
            if iteration > 5:
                if multi_machine_improvement < 0.05:
                    print("Slow multi-machine improvement, boosting exploration...")
                    hyperparams = hyperparams.copy()
                    hyperparams['ent_coef'] = min(0.25, hyperparams['ent_coef'] * 1.3)
                
                if dependency_handling_improvement < 0.03:
                    print("Poor dependency handling, increasing learning epochs...")
                    hyperparams = hyperparams.copy()
                    hyperparams['n_epochs'] = min(35, hyperparams['n_epochs'] + 5)
            
            print(f"Phase: {hyperparams['phase']}")
            print(f"Hyperparameters: LR={hyperparams['learning_rate']:.1e}, "
                  f"Entropy={hyperparams['ent_coef']}, "
                  f"Clip={hyperparams['clip_range']}, "
                  f"Epochs={hyperparams['n_epochs']}")
            
            # Create or update model
            if model is None:
                model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=hyperparams['learning_rate'],
                    n_steps=2048,  # Larger for complex scenarios
                    batch_size=hyperparams['batch_size'],
                    n_epochs=hyperparams['n_epochs'],
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=hyperparams['clip_range'],
                    ent_coef=hyperparams['ent_coef'],
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=dict(
                            pi=[512, 512, 256],  # Deeper network for complexity
                            vf=[512, 512, 256]
                        )
                    ),
                    verbose=1
                )
            else:
                # Update learning rate and entropy
                model.learning_rate = hyperparams['learning_rate']
                model.ent_coef = hyperparams['ent_coef']
            
            # Set up checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=100000,
                save_path=os.path.join(self.output_dir, "checkpoints"),
                name_prefix=f"complex_iter{iteration+1}"
            )
            
            # Train
            print(f"Training for {timesteps_per_iter:,} timesteps...")
            model.learn(
                total_timesteps=timesteps_per_iter, 
                reset_num_timesteps=False,
                callback=checkpoint_callback
            )
            total_timesteps += timesteps_per_iter
            
            # Evaluate
            print("Evaluating complex scenario handling...")
            avg_completion, results = self.evaluate_model(model, n_episodes=30)  # More episodes for complex
            avg_reward = np.mean(results['rewards'])
            avg_on_time = np.mean(results['on_time_rates'])
            avg_utilization = np.mean(results['utilizations'])
            avg_late_penalty = np.mean(results['late_penalties'])
            
            # Complex-specific metrics
            total_multi_machine = sum(results['multi_machine_jobs_completed'])
            total_violations = sum(results['dependency_violations'])
            total_conflicts = sum(results['resource_conflicts'])
            avg_complexity = np.mean(results['avg_job_complexity'])
            
            multi_machine_rate = total_multi_machine / (30 * 5)  # Assuming ~5 multi-machine jobs per episode
            violation_rate = total_violations / (30 * 20)  # Normalized by episodes and jobs
            conflict_rate = total_conflicts / (30 * 20)
            
            iteration_time = time.time() - start_time
            
            print(f"\nResults:")
            print(f"  Completion Rate: {avg_completion*100:.1f}%")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  On-time Rate: {avg_on_time*100:.1f}%")
            print(f"  Utilization: {avg_utilization*100:.1f}%")
            print(f"  Multi-Machine Success Rate: {multi_machine_rate*100:.1f}%")
            print(f"  Dependency Violation Rate: {violation_rate*100:.1f}%")
            print(f"  Resource Conflict Rate: {conflict_rate*100:.1f}%")
            print(f"  Average Job Complexity: {avg_complexity:.2f}")
            print(f"  Late Penalty: {avg_late_penalty:.2f}")
            print(f"  Iteration Time: {iteration_time/60:.1f} minutes")
            
            # Calculate improvements
            if len(history['multi_machine_success_rate']) > 0:
                multi_machine_improvement = multi_machine_rate - history['multi_machine_success_rate'][-1]
                dependency_handling_improvement = history['dependency_violation_rate'][-1] - violation_rate
                print(f"  Multi-Machine Improvement: {multi_machine_improvement*100:.1f}%")
                print(f"  Dependency Handling Improvement: {dependency_handling_improvement*100:.1f}%")
            
            # Save if best
            if avg_completion > best_completion:
                best_completion = avg_completion
                model_path = os.path.join(
                    self.output_dir, 
                    f"complex_best_model_{avg_completion*100:.0f}pct.zip"
                )
                model.save(model_path)
                history['best_model_path'] = model_path
                history['best_completion'] = best_completion
                print(f"  >> New best model saved! ({best_completion*100:.1f}%)")
            
            # Record history
            history['iterations'].append(iteration + 1)
            history['completion_rates'].append(avg_completion)
            history['rewards'].append(avg_reward)
            history['on_time_rates'].append(avg_on_time)
            history['utilizations'].append(avg_utilization)
            history['late_penalties'].append(avg_late_penalty)
            history['multi_machine_success_rate'].append(multi_machine_rate)
            history['dependency_violation_rate'].append(violation_rate)
            history['resource_conflict_rate'].append(conflict_rate)
            history['learning_rates'].append(hyperparams['learning_rate'])
            history['entropy_coeffs'].append(hyperparams['ent_coef'])
            history['training_phases'].append(hyperparams['phase'])
            
            # Save intermediate results
            history_path = os.path.join(self.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Check if target reached
            if avg_completion >= target:
                print(f"\n✓ TARGET REACHED! {avg_completion*100:.1f}% >= {target*100:.0f}%")
                print(f"Total timesteps: {total_timesteps:,}")
                break
            
            # Complex-specific strategy adjustments
            if iteration > 4:
                # If multi-machine jobs are failing
                if multi_machine_rate < 0.5:
                    print("\nPoor multi-machine coordination, increasing network capacity...")
                    model.policy_kwargs['net_arch']['pi'].append(128)
                    model.policy_kwargs['net_arch']['vf'].append(128)
                
                # If too many dependency violations
                if violation_rate > 0.2:
                    print("High dependency violations, slowing learning for stability...")
                    model.learning_rate = model.learning_rate * 0.8
                
                # If resource conflicts are high
                if conflict_rate > 0.15:
                    print("Resource conflicts detected, improving exploration...")
                    model.ent_coef = min(0.3, model.ent_coef * 1.2)
        
        env.close()
        
        # Final summary
        final_result = {
            'strategy': 'small_complex',
            'final_completion': best_completion,
            'target_reached': best_completion >= target,
            'iterations_used': len(history['iterations']),
            'total_timesteps': total_timesteps,
            'best_model_path': history['best_model_path'],
            'final_metrics': {
                'completion_rate': best_completion,
                'on_time_rate': history['on_time_rates'][-1] if history['on_time_rates'] else 0,
                'utilization': history['utilizations'][-1] if history['utilizations'] else 0,
                'multi_machine_success_rate': history['multi_machine_success_rate'][-1] if history['multi_machine_success_rate'] else 0,
                'dependency_violation_rate': history['dependency_violation_rate'][-1] if history['dependency_violation_rate'] else 0,
                'resource_conflict_rate': history['resource_conflict_rate'][-1] if history['resource_conflict_rate'] else 0,
                'late_penalty': history['late_penalties'][-1] if history['late_penalties'] else 0
            },
            'history': history
        }
        
        # Save final results
        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\n{'='*70}")
        print("COMPLEX STRATEGY TRAINING COMPLETE")
        print(f"Final Completion Rate: {best_completion*100:.1f}%")
        print(f"Target Reached: {'YES' if final_result['target_reached'] else 'NO'}")
        print(f"Multi-Machine Success: {final_result['final_metrics']['multi_machine_success_rate']*100:.1f}%")
        print(f"Dependency Violations: {final_result['final_metrics']['dependency_violation_rate']*100:.1f}%")
        print(f"Resource Conflicts: {final_result['final_metrics']['resource_conflict_rate']*100:.1f}%")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
        
        return final_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Small Complex strategy to 80% completion")
    parser.add_argument('--output-dir', type=str, 
                       default="/home/azureuser/ppo/app_2/phase4/results/complex",
                       help='Output directory for results')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    trainer = ComplexStrategyTrainer(output_dir=args.output_dir)
    
    if args.eval_only and args.model_path:
        # Load and evaluate existing model
        print(f"Evaluating model: {args.model_path}")
        model = PPO.load(args.model_path)
        avg_completion, results = trainer.evaluate_model(model, n_episodes=50)
        
        print(f"\nEvaluation Results (50 episodes):")
        print(f"  Completion Rate: {avg_completion*100:.1f}%")
        print(f"  Average Reward: {np.mean(results['rewards']):.2f}")
        print(f"  On-time Rate: {np.mean(results['on_time_rates'])*100:.1f}%")
        print(f"  Utilization: {np.mean(results['utilizations'])*100:.1f}%")
        print(f"  Multi-Machine Jobs: {sum(results['multi_machine_jobs_completed'])}")
        print(f"  Dependency Violations: {sum(results['dependency_violations'])}")
        print(f"  Resource Conflicts: {sum(results['resource_conflicts'])}")
    else:
        # Train to target
        result = trainer.train_to_target()