#!/usr/bin/env python
"""
Evaluation script for trained PPO scheduling models.
Calculates comprehensive metrics and compares against baselines.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.scheduling_env import SchedulingEnv
from models.ppo_scheduler import PPOScheduler


class ModelEvaluator:
    """Evaluate trained PPO models with comprehensive metrics."""
    
    def __init__(self, data_path: str, model_path: str = None, device: str = "mps"):
        """
        Initialize evaluator.
        
        Args:
            data_path: Path to JSON data file
            model_path: Path to trained model checkpoint
            device: Compute device (cpu, cuda, mps)
        """
        self.data_path = data_path
        self.model_path = model_path
        self.device = device
        
        # Create environment
        self.env = SchedulingEnv(data_path, max_steps=5000)
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = PPOScheduler(
                obs_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                device=device
            )
            self.model.load(model_path)
            print(f"Loaded model from {model_path}")
    
    def evaluate_episode(self, deterministic: bool = True) -> Dict:
        """
        Run one evaluation episode.
        
        Args:
            deterministic: Use deterministic policy
            
        Returns:
            Dictionary of metrics
        """
        obs, info = self.env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 5000:
            if self.model:
                action, _ = self.model.predict(obs, info['action_mask'], deterministic)
            else:
                # Random valid action for baseline
                valid_actions = np.where(info['action_mask'])[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Calculate metrics
        metrics = self._calculate_metrics(info)
        metrics['total_reward'] = total_reward
        metrics['steps'] = steps
        
        return metrics
    
    def _calculate_metrics(self, final_info: Dict) -> Dict:
        """
        Calculate comprehensive metrics from final episode info.
        
        Args:
            final_info: Final info dict from environment
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic completion metrics
        metrics['tasks_scheduled'] = final_info.get('tasks_scheduled', 0)
        metrics['total_tasks'] = final_info.get('total_tasks', 0)
        metrics['completion_rate'] = (
            metrics['tasks_scheduled'] / metrics['total_tasks'] 
            if metrics['total_tasks'] > 0 else 0
        )
        
        # Constraint satisfaction
        metrics['constraint_violations'] = len(final_info.get('constraint_violations', []))
        metrics['constraint_satisfaction_rate'] = (
            1.0 if metrics['tasks_scheduled'] == 0 
            else 1.0 - (metrics['constraint_violations'] / metrics['tasks_scheduled'])
        )
        
        # Timing metrics
        if hasattr(self.env, 'task_schedules'):
            schedules = self.env.task_schedules
            if schedules:
                # Makespan (total time to complete all tasks)
                end_times = [end for _, end, _ in schedules.values()]
                metrics['makespan'] = max(end_times) if end_times else 0
                
                # Machine utilization
                metrics['machine_utilization'] = self._calculate_utilization()
                
                # On-time delivery
                metrics['on_time_rate'] = self._calculate_on_time_rate()
                
                # Average flow time
                start_times = [start for start, _, _ in schedules.values()]
                metrics['avg_flow_time'] = (
                    np.mean([end - start for (start, end, _) in schedules.values()])
                    if schedules else 0
                )
        else:
            metrics['makespan'] = 0
            metrics['machine_utilization'] = 0
            metrics['on_time_rate'] = 0
            metrics['avg_flow_time'] = 0
        
        return metrics
    
    def _calculate_utilization(self) -> float:
        """
        Calculate average machine utilization.
        
        Returns:
            Utilization percentage (0-1)
        """
        if not hasattr(self.env, 'machine_schedules'):
            return 0
            
        total_busy_time = 0
        total_available_time = 0
        
        for machine, schedule in self.env.machine_schedules.items():
            if schedule:
                # Calculate busy time for this machine
                busy_time = sum(end - start for start, end in schedule)
                total_busy_time += busy_time
                
                # Available time is makespan
                max_end = max(end for _, end in schedule)
                total_available_time += max_end
        
        if total_available_time > 0:
            return total_busy_time / total_available_time
        return 0
    
    def _calculate_on_time_rate(self) -> float:
        """
        Calculate percentage of tasks completed on time.
        
        Returns:
            On-time rate (0-1)
        """
        if not hasattr(self.env, 'task_schedules'):
            return 0
            
        on_time_count = 0
        total_count = 0
        
        for task_idx, (start, end, machine) in self.env.task_schedules.items():
            task = self.env.loader.tasks[task_idx]
            family = self.env.loader.families[task.family_id]
            
            # Check if completed before LCD
            lcd_hours = family.lcd_days_remaining * 24
            if end <= lcd_hours:
                on_time_count += 1
            total_count += 1
        
        if total_count > 0:
            return on_time_count / total_count
        return 0
    
    def evaluate_multiple(self, n_episodes: int = 10) -> Dict:
        """
        Run multiple evaluation episodes and aggregate results.
        
        Args:
            n_episodes: Number of episodes to run
            
        Returns:
            Aggregated metrics dictionary
        """
        all_metrics = []
        
        for episode in range(n_episodes):
            metrics = self.evaluate_episode()
            all_metrics.append(metrics)
            
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Completion: {metrics['completion_rate']*100:.1f}%, "
                  f"Makespan: {metrics['makespan']:.1f}h")
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def print_summary(self, metrics: Dict):
        """
        Print formatted summary of metrics.
        
        Args:
            metrics: Dictionary of metrics to print
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Extract mean values for key metrics
        if 'completion_rate_mean' in metrics:
            print(f"Completion Rate: {metrics['completion_rate_mean']*100:.1f}% "
                  f"(±{metrics['completion_rate_std']*100:.1f}%)")
            print(f"Constraint Satisfaction: {metrics['constraint_satisfaction_rate_mean']*100:.1f}%")
            print(f"On-Time Delivery: {metrics['on_time_rate_mean']*100:.1f}%")
            print(f"Machine Utilization: {metrics['machine_utilization_mean']*100:.1f}%")
            print(f"Average Makespan: {metrics['makespan_mean']:.1f} hours")
            print(f"Average Flow Time: {metrics['avg_flow_time_mean']:.1f} hours")
        else:
            # Single episode metrics
            print(f"Completion Rate: {metrics['completion_rate']*100:.1f}%")
            print(f"Tasks Scheduled: {metrics['tasks_scheduled']}/{metrics['total_tasks']}")
            print(f"Constraint Violations: {metrics['constraint_violations']}")
            print(f"On-Time Delivery: {metrics['on_time_rate']*100:.1f}%")
            print(f"Machine Utilization: {metrics['machine_utilization']*100:.1f}%")
            print(f"Makespan: {metrics['makespan']:.1f} hours")
        
        print("="*60)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained PPO models')
    parser.add_argument('--data', type=str, default='data/40_jobs.json',
                       help='Path to test data')
    parser.add_argument('--model', type=str, default='checkpoints/fast/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='mps',
                       help='Compute device (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.data, args.model, args.device)
    
    # Run evaluation
    if args.episodes > 1:
        metrics = evaluator.evaluate_multiple(args.episodes)
    else:
        metrics = evaluator.evaluate_episode()
    
    # Print results
    evaluator.print_summary(metrics)
    
    # Save results
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'eval_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()