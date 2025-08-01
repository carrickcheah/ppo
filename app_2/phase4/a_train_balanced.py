"""
Train Small Balanced Environment to 80% completion rate
Balanced workload - easiest to achieve 80%
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

from environments.small_balanced_env import SmallBalancedEnvironment


class BalancedStrategyTrainer:
    """Trains Small Balanced strategy until 80% completion rate is achieved."""
    
    def __init__(self, output_dir: str = "/home/azureuser/ppo/app_2/phase4/results/balanced"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Strategy configuration
        self.config = {
            'env_class': SmallBalancedEnvironment,
            'target_completion': 0.80,
            'max_iterations': 10,
            'timesteps_per_iteration': 200000,
            'description': 'Balanced workload - easiest to achieve 80%'
        }
        
        # Progressive PPO hyperparameters for balanced strategy
        self.hyperparameter_schedules = [
            # Initial exploration (iterations 1-3)
            {
                'learning_rate': 5e-4,
                'ent_coef': 0.1,
                'clip_range': 0.3,
                'n_epochs': 10,
                'batch_size': 64,
                'phase': 'exploration'
            },
            # Balanced learning (iterations 4-6)
            {
                'learning_rate': 3e-4,
                'ent_coef': 0.05,
                'clip_range': 0.2,
                'n_epochs': 15,
                'batch_size': 128,
                'phase': 'balanced'
            },
            # Fine-tuning (iterations 7+)
            {
                'learning_rate': 1e-4,
                'ent_coef': 0.02,
                'clip_range': 0.15,
                'n_epochs': 20,
                'batch_size': 256,
                'phase': 'fine-tuning'
            }
        ]
    
    def evaluate_model(self, model, n_episodes: int = 10) -> tuple[float, dict]:
        """Evaluate model performance on balanced environment."""
        env = self.config['env_class'](verbose=False)
        
        results = {
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': [],
            'late_penalties': []
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
            
            late_penalty = info.get('total_late_penalty', 0)
            results['late_penalties'].append(late_penalty)
        
        env.close()
        
        avg_completion = np.mean(results['completion_rates'])
        return avg_completion, results
    
    def train_to_target(self) -> dict:
        """Train balanced strategy until target completion rate is achieved."""
        target = self.config['target_completion']
        max_iterations = self.config['max_iterations']
        timesteps_per_iter = self.config['timesteps_per_iteration']
        
        print(f"\n{'='*70}")
        print(f"Training SMALL BALANCED STRATEGY to {target*100:.0f}% completion")
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
            'strategy': 'small_balanced',
            'iterations': [],
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': [],
            'late_penalties': [],
            'learning_rates': [],
            'entropy_coeffs': [],
            'best_completion': 0,
            'best_model_path': None,
            'training_phases': []
        }
        
        model = None
        best_completion = 0
        total_timesteps = 0
        
        for iteration in range(max_iterations):
            start_time = time.time()
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Select hyperparameters based on progress
            hp_index = min(iteration // 3, len(self.hyperparameter_schedules) - 1)
            hyperparams = self.hyperparameter_schedules[hp_index]
            
            print(f"Phase: {hyperparams['phase']}")
            print(f"Hyperparameters: LR={hyperparams['learning_rate']}, "
                  f"Entropy={hyperparams['ent_coef']}, "
                  f"Clip={hyperparams['clip_range']}")
            
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
            
            # Set up checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=os.path.join(self.output_dir, "checkpoints"),
                name_prefix=f"balanced_iter{iteration+1}"
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
            print("Evaluating performance...")
            avg_completion, results = self.evaluate_model(model, n_episodes=20)
            avg_reward = np.mean(results['rewards'])
            avg_on_time = np.mean(results['on_time_rates'])
            avg_utilization = np.mean(results['utilizations'])
            avg_late_penalty = np.mean(results['late_penalties'])
            
            iteration_time = time.time() - start_time
            
            print(f"\nResults:")
            print(f"  Completion Rate: {avg_completion*100:.1f}%")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  On-time Rate: {avg_on_time*100:.1f}%")
            print(f"  Utilization: {avg_utilization*100:.1f}%")
            print(f"  Late Penalty: {avg_late_penalty:.2f}")
            print(f"  Iteration Time: {iteration_time/60:.1f} minutes")
            
            # Save if best
            if avg_completion > best_completion:
                best_completion = avg_completion
                model_path = os.path.join(
                    self.output_dir, 
                    f"balanced_best_model_{avg_completion*100:.0f}pct.zip"
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
            
            # Adjust strategy if stuck
            if iteration > 2 and len(history['completion_rates']) >= 3:
                recent_rates = history['completion_rates'][-3:]
                if max(recent_rates) - min(recent_rates) < 0.02:
                    print("\nPerformance plateaued, adjusting exploration...")
                    model.ent_coef = min(0.2, model.ent_coef * 1.5)
                    print(f"Increased entropy coefficient to {model.ent_coef}")
        
        env.close()
        
        # Final summary
        final_result = {
            'strategy': 'small_balanced',
            'final_completion': best_completion,
            'target_reached': best_completion >= target,
            'iterations_used': len(history['iterations']),
            'total_timesteps': total_timesteps,
            'best_model_path': history['best_model_path'],
            'final_metrics': {
                'completion_rate': best_completion,
                'on_time_rate': history['on_time_rates'][-1] if history['on_time_rates'] else 0,
                'utilization': history['utilizations'][-1] if history['utilizations'] else 0,
                'late_penalty': history['late_penalties'][-1] if history['late_penalties'] else 0
            },
            'history': history
        }
        
        # Save final results
        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\n{'='*70}")
        print("BALANCED STRATEGY TRAINING COMPLETE")
        print(f"Final Completion Rate: {best_completion*100:.1f}%")
        print(f"Target Reached: {'YES' if final_result['target_reached'] else 'NO'}")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
        
        return final_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Small Balanced strategy to 80% completion")
    parser.add_argument('--output-dir', type=str, 
                       default="/home/azureuser/ppo/app_2/phase4/results/balanced",
                       help='Output directory for results')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    trainer = BalancedStrategyTrainer(output_dir=args.output_dir)
    
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
    else:
        # Train to target
        result = trainer.train_to_target()