#!/usr/bin/env python
"""
Baseline scheduling algorithms for comparison with PPO model.
Implements FIFO, EDD, SPT, and Random schedulers.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.scheduling_env import SchedulingEnv


class BaselineScheduler(ABC):
    """Abstract base class for baseline schedulers."""
    
    def __init__(self, env: SchedulingEnv):
        """
        Initialize baseline scheduler.
        
        Args:
            env: Scheduling environment
        """
        self.env = env
        self.name = "BaseScheduler"
    
    @abstractmethod
    def select_task(self, valid_tasks: List[int]) -> int:
        """
        Select next task to schedule.
        
        Args:
            valid_tasks: List of valid task indices
            
        Returns:
            Selected task index
        """
        pass
    
    def schedule(self) -> Dict:
        """
        Run complete scheduling episode.
        
        Returns:
            Dictionary of metrics
        """
        obs, info = self.env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 5000:
            # Get valid tasks
            valid_tasks = np.where(info['action_mask'])[0]
            
            if len(valid_tasks) == 0:
                break  # No valid actions
            
            # Select task using baseline strategy
            action = self.select_task(valid_tasks)
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Return metrics
        return {
            'scheduler': self.name,
            'tasks_scheduled': info['tasks_scheduled'],
            'total_tasks': info['total_tasks'],
            'completion_rate': info['tasks_scheduled'] / info['total_tasks'],
            'total_reward': total_reward,
            'steps': steps
        }


class FIFOScheduler(BaselineScheduler):
    """First In First Out scheduler - schedules tasks in order."""
    
    def __init__(self, env: SchedulingEnv):
        super().__init__(env)
        self.name = "FIFO"
    
    def select_task(self, valid_tasks: List[int]) -> int:
        """Select first valid task (lowest index)."""
        return valid_tasks[0]


class EDDScheduler(BaselineScheduler):
    """Earliest Due Date scheduler - prioritizes tasks with earliest LCD."""
    
    def __init__(self, env: SchedulingEnv):
        super().__init__(env)
        self.name = "EDD"
    
    def select_task(self, valid_tasks: List[int]) -> int:
        """Select task with earliest due date (LCD)."""
        earliest_lcd = float('inf')
        selected_task = valid_tasks[0]
        
        for task_idx in valid_tasks:
            task = self.env.loader.tasks[task_idx]
            family = self.env.loader.families[task.family_id]
            lcd_days = family.lcd_days_remaining
            
            if lcd_days < earliest_lcd:
                earliest_lcd = lcd_days
                selected_task = task_idx
        
        return selected_task


class SPTScheduler(BaselineScheduler):
    """Shortest Processing Time scheduler - prioritizes quick tasks."""
    
    def __init__(self, env: SchedulingEnv):
        super().__init__(env)
        self.name = "SPT"
    
    def select_task(self, valid_tasks: List[int]) -> int:
        """Select task with shortest processing time."""
        shortest_time = float('inf')
        selected_task = valid_tasks[0]
        
        for task_idx in valid_tasks:
            task = self.env.loader.tasks[task_idx]
            
            if task.processing_time < shortest_time:
                shortest_time = task.processing_time
                selected_task = task_idx
        
        return selected_task


class RandomScheduler(BaselineScheduler):
    """Random scheduler - randomly selects from valid tasks."""
    
    def __init__(self, env: SchedulingEnv):
        super().__init__(env)
        self.name = "Random"
    
    def select_task(self, valid_tasks: List[int]) -> int:
        """Select random valid task."""
        return np.random.choice(valid_tasks)


class LPTScheduler(BaselineScheduler):
    """Longest Processing Time scheduler - prioritizes longer tasks first."""
    
    def __init__(self, env: SchedulingEnv):
        super().__init__(env)
        self.name = "LPT"
    
    def select_task(self, valid_tasks: List[int]) -> int:
        """Select task with longest processing time."""
        longest_time = -1
        selected_task = valid_tasks[0]
        
        for task_idx in valid_tasks:
            task = self.env.loader.tasks[task_idx]
            
            if task.processing_time > longest_time:
                longest_time = task.processing_time
                selected_task = task_idx
        
        return selected_task


class CriticalRatioScheduler(BaselineScheduler):
    """Critical Ratio scheduler - prioritizes based on time remaining vs processing time."""
    
    def __init__(self, env: SchedulingEnv):
        super().__init__(env)
        self.name = "CR"
    
    def select_task(self, valid_tasks: List[int]) -> int:
        """Select task with lowest critical ratio (time_remaining / processing_time)."""
        best_ratio = float('inf')
        selected_task = valid_tasks[0]
        
        for task_idx in valid_tasks:
            task = self.env.loader.tasks[task_idx]
            family = self.env.loader.families[task.family_id]
            
            # Calculate time remaining until LCD
            time_remaining = family.lcd_days_remaining * 24 - self.env.current_time
            
            # Critical ratio (lower is more critical)
            if task.processing_time > 0:
                ratio = time_remaining / task.processing_time
            else:
                ratio = float('inf')
            
            if ratio < best_ratio:
                best_ratio = ratio
                selected_task = task_idx
        
        return selected_task


def compare_baselines(data_path: str, n_episodes: int = 5) -> Dict:
    """
    Compare all baseline schedulers.
    
    Args:
        data_path: Path to test data
        n_episodes: Number of episodes per baseline
        
    Returns:
        Comparison results dictionary
    """
    # Create environment
    env = SchedulingEnv(data_path, max_steps=5000)
    
    # Initialize all baselines
    baselines = [
        FIFOScheduler(env),
        EDDScheduler(env),
        SPTScheduler(env),
        LPTScheduler(env),
        CriticalRatioScheduler(env),
        RandomScheduler(env)
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    for baseline in baselines:
        print(f"\nTesting {baseline.name} scheduler...")
        
        episode_results = []
        for episode in range(n_episodes):
            metrics = baseline.schedule()
            episode_results.append(metrics)
            
            print(f"  Episode {episode + 1}: {metrics['tasks_scheduled']}/{metrics['total_tasks']} "
                  f"({metrics['completion_rate']*100:.1f}%)")
        
        # Aggregate results
        results[baseline.name] = {
            'completion_rate_mean': np.mean([r['completion_rate'] for r in episode_results]),
            'completion_rate_std': np.std([r['completion_rate'] for r in episode_results]),
            'tasks_scheduled_mean': np.mean([r['tasks_scheduled'] for r in episode_results]),
            'total_reward_mean': np.mean([r['total_reward'] for r in episode_results]),
            'steps_mean': np.mean([r['steps'] for r in episode_results])
        }
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Scheduler':<15} {'Completion Rate':<20} {'Avg Tasks':<15} {'Avg Steps':<10}")
    print("-"*60)
    
    for name, metrics in results.items():
        completion = f"{metrics['completion_rate_mean']*100:.1f}% (±{metrics['completion_rate_std']*100:.1f}%)"
        print(f"{name:<15} {completion:<20} {metrics['tasks_scheduled_mean']:<15.1f} {metrics['steps_mean']:<10.0f}")
    
    # Find best baseline
    best_baseline = max(results.items(), key=lambda x: x[1]['completion_rate_mean'])
    print(f"\nBest baseline: {best_baseline[0]} with {best_baseline[1]['completion_rate_mean']*100:.1f}% completion")
    
    return results


def main():
    """Main function for baseline comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline schedulers')
    parser.add_argument('--data', type=str, default='data/40_jobs.json',
                       help='Path to test data')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes per baseline')
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_baselines(args.data, args.episodes)
    
    # Save results
    import json
    from datetime import datetime
    
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'baselines_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()