"""
Train Improved Bottleneck Environment with curriculum learning
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

from environments.improved_bottleneck_env import ImprovedBottleneckEnvironment


class ImprovedBottleneckTrainer:
    """Trains Improved Bottleneck strategy with curriculum learning."""
    
    def __init__(self, output_dir: str = "/home/azureuser/ppo/app_2/phase4/results/improved_bottleneck"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Strategy configuration
        self.config = {
            'env_class': ImprovedBottleneckEnvironment,
            'target_completion': 0.80,
            'max_iterations': 15,
            'timesteps_per_iteration': 300000,
            'description': 'Improved bottleneck with action masking and better rewards'
        }
        
        # Curriculum learning stages
        self.curriculum_stages = [
            {
                'name': 'exploration',
                'iterations': 3,
                'learning_rate': 1e-3,
                'ent_coef': 0.2,  # High entropy for exploration
                'clip_range': 0.3,
                'n_epochs': 10,
                'batch_size': 64,
                'n_steps': 512,
            },
            {
                'name': 'learning',
                'iterations': 5,
                'learning_rate': 5e-4,
                'ent_coef': 0.1,
                'clip_range': 0.25,
                'n_epochs': 15,
                'batch_size': 128,
                'n_steps': 1024,
            },
            {
                'name': 'refinement',
                'iterations': 5,
                'learning_rate': 2e-4,
                'ent_coef': 0.05,
                'clip_range': 0.2,
                'n_epochs': 20,
                'batch_size': 256,
                'n_steps': 2048,
            },
            {
                'name': 'fine_tuning',
                'iterations': 2,
                'learning_rate': 1e-4,
                'ent_coef': 0.02,
                'clip_range': 0.15,
                'n_epochs': 25,
                'batch_size': 256,
                'n_steps': 2048,
            }
        ]
    
    def evaluate_model(self, model, n_episodes: int = 20, deterministic: bool = False) -> tuple[float, dict]:
        """Evaluate model performance with option for stochastic evaluation."""
        env = self.config['env_class'](verbose=False)
        
        results = {
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': [],
            'late_penalties': [],
            'throughputs': [],
            'episode_lengths': [],
            'invalid_actions': []
        }
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            invalid_count = 0
            
            while not done:
                # Use deterministic or stochastic policy
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                # Track invalid actions
                if info.get('action_type') == 'invalid':
                    invalid_count += 1
            
            # Extract metrics
            completion_rate = info.get('completion_rate', 0)
            results['completion_rates'].append(completion_rate)
            results['rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)
            results['invalid_actions'].append(invalid_count)
            
            # Calculate on-time rate properly
            scheduled = info.get('scheduled_jobs', 0)
            on_time = info.get('on_time_jobs', 0)
            on_time_rate = on_time / max(scheduled, 1)
            results['on_time_rates'].append(on_time_rate)
            
            utilization = info.get('machine_utilization', 0)
            results['utilizations'].append(utilization)
            
            late_penalty = info.get('total_late_penalty', 0)
            results['late_penalties'].append(late_penalty)
            
            throughput = info.get('throughput_per_day', 0)
            results['throughputs'].append(throughput)
        
        env.close()
        
        avg_completion = np.mean(results['completion_rates'])
        return avg_completion, results
    
    def train_to_target(self) -> dict:
        """Train improved bottleneck strategy with curriculum learning."""
        target = self.config['target_completion']
        max_iterations = self.config['max_iterations']
        timesteps_per_iter = self.config['timesteps_per_iteration']
        
        print(f"\n{'='*70}")
        print(f"Training IMPROVED BOTTLENECK STRATEGY")
        print(f"Target: {target*100:.0f}% completion")
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
            'strategy': 'improved_bottleneck',
            'iterations': [],
            'completion_rates': [],
            'rewards': [],
            'on_time_rates': [],
            'utilizations': [],
            'throughputs': [],
            'invalid_action_rates': [],
            'learning_rates': [],
            'entropy_coeffs': [],
            'best_completion': 0,
            'best_model_path': None,
            'curriculum_stages': []
        }
        
        model = None
        best_completion = 0
        total_timesteps = 0
        current_stage_idx = 0
        stage_iteration = 0
        
        for iteration in range(max_iterations):
            start_time = time.time()
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Get current curriculum stage
            if stage_iteration >= self.curriculum_stages[current_stage_idx]['iterations']:
                if current_stage_idx < len(self.curriculum_stages) - 1:
                    current_stage_idx += 1
                    stage_iteration = 0
            
            stage = self.curriculum_stages[current_stage_idx]
            stage_iteration += 1
            
            print(f"Curriculum Stage: {stage['name']} (iteration {stage_iteration}/{stage['iterations']})")
            print(f"Hyperparameters: LR={stage['learning_rate']:.1e}, "
                  f"Entropy={stage['ent_coef']}, Clip={stage['clip_range']}")
            
            # Create or update model
            if model is None:
                model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=stage['learning_rate'],
                    n_steps=stage['n_steps'],
                    batch_size=stage['batch_size'],
                    n_epochs=stage['n_epochs'],
                    gamma=0.995,  # Slightly higher for long episodes
                    gae_lambda=0.98,
                    clip_range=stage['clip_range'],
                    ent_coef=stage['ent_coef'],
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=dict(
                            pi=[512, 512, 256],  # Larger network for complex task
                            vf=[512, 512, 256]
                        ),
                        activation_fn=nn.ReLU
                    ),
                    verbose=1
                )
            else:
                # Update hyperparameters
                model.learning_rate = stage['learning_rate']
                model.ent_coef = stage['ent_coef']
                model.clip_range = stage['clip_range']
                model.n_epochs = stage['n_epochs']
            
            # Set up checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=os.path.join(self.output_dir, "checkpoints"),
                name_prefix=f"improved_bottleneck_iter{iteration+1}"
            )
            
            # Train
            print(f"Training for {timesteps_per_iter:,} timesteps...")
            model.learn(
                total_timesteps=timesteps_per_iter, 
                reset_num_timesteps=False,
                callback=checkpoint_callback
            )
            total_timesteps += timesteps_per_iter
            
            # Evaluate with both deterministic and stochastic
            print("\nEvaluating performance...")
            avg_completion_det, results_det = self.evaluate_model(model, n_episodes=10, deterministic=True)
            avg_completion_stoch, results_stoch = self.evaluate_model(model, n_episodes=10, deterministic=False)
            
            # Use better of the two
            if avg_completion_stoch > avg_completion_det:
                print("Using stochastic evaluation results (better performance)")
                avg_completion = avg_completion_stoch
                results = results_stoch
            else:
                avg_completion = avg_completion_det
                results = results_det
            
            # Calculate metrics
            avg_reward = np.mean(results['rewards'])
            avg_on_time = np.mean(results['on_time_rates'])
            avg_utilization = np.mean(results['utilizations'])
            avg_throughput = np.mean(results['throughputs'])
            avg_episode_length = np.mean(results['episode_lengths'])
            avg_invalid_actions = np.mean(results['invalid_actions'])
            invalid_rate = avg_invalid_actions / avg_episode_length if avg_episode_length > 0 else 0
            
            iteration_time = time.time() - start_time
            
            print(f"\nResults:")
            print(f"  Completion Rate: {avg_completion*100:.1f}%")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  On-time Rate: {avg_on_time*100:.1f}%")
            print(f"  Machine Utilization: {avg_utilization*100:.1f}%")
            print(f"  Throughput: {avg_throughput:.1f} jobs/day")
            print(f"  Invalid Action Rate: {invalid_rate*100:.1f}%")
            print(f"  Episode Length: {avg_episode_length:.0f} steps")
            print(f"  Iteration Time: {iteration_time/60:.1f} minutes")
            
            # Save if best
            if avg_completion > best_completion:
                best_completion = avg_completion
                model_path = os.path.join(
                    self.output_dir, 
                    f"improved_bottleneck_best_{avg_completion*100:.0f}pct.zip"
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
            history['throughputs'].append(avg_throughput)
            history['invalid_action_rates'].append(invalid_rate)
            history['learning_rates'].append(stage['learning_rate'])
            history['entropy_coeffs'].append(stage['ent_coef'])
            history['curriculum_stages'].append(stage['name'])
            
            # Save intermediate results
            history_path = os.path.join(self.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Check if target reached
            if avg_completion >= target:
                print(f"\n✓ TARGET REACHED! {avg_completion*100:.1f}% >= {target*100:.0f}%")
                print(f"Total timesteps: {total_timesteps:,}")
                break
            
            # Early stopping if no improvement
            if iteration > 5:
                recent_completions = history['completion_rates'][-5:]
                if max(recent_completions) - min(recent_completions) < 0.02:
                    print("\nNo significant improvement in last 5 iterations.")
                    if current_stage_idx < len(self.curriculum_stages) - 1:
                        print("Moving to next curriculum stage...")
                        current_stage_idx += 1
                        stage_iteration = 0
        
        env.close()
        
        # Final evaluation with more episodes
        print("\nFinal evaluation (50 episodes)...")
        model = PPO.load(history['best_model_path'])
        final_completion, final_results = self.evaluate_model(model, n_episodes=50)
        
        # Final summary
        final_result = {
            'strategy': 'improved_bottleneck',
            'final_completion': final_completion,
            'target_reached': final_completion >= target,
            'iterations_used': len(history['iterations']),
            'total_timesteps': total_timesteps,
            'best_model_path': history['best_model_path'],
            'final_metrics': {
                'completion_rate': final_completion,
                'avg_reward': np.mean(final_results['rewards']),
                'on_time_rate': np.mean(final_results['on_time_rates']),
                'utilization': np.mean(final_results['utilizations']),
                'throughput': np.mean(final_results['throughputs']),
                'invalid_action_rate': np.mean([i/l for i, l in zip(final_results['invalid_actions'], final_results['episode_lengths'])])
            },
            'history': history
        }
        
        # Save final results
        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\n{'='*70}")
        print("IMPROVED BOTTLENECK TRAINING COMPLETE")
        print(f"Final Completion Rate: {final_completion*100:.1f}%")
        print(f"Target Reached: {'YES' if final_result['target_reached'] else 'NO'}")
        print(f"Machine Utilization: {final_result['final_metrics']['utilization']*100:.1f}%")
        print(f"Throughput: {final_result['final_metrics']['throughput']:.1f} jobs/day")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
        
        return final_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Improved Bottleneck strategy")
    parser.add_argument('--output-dir', type=str, 
                       default="/home/azureuser/ppo/app_2/phase4/results/improved_bottleneck",
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    trainer = ImprovedBottleneckTrainer(output_dir=args.output_dir)
    result = trainer.train_to_target()