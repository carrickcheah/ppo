#!/usr/bin/env python
"""
Evaluate the 10x improved model performance
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler

def evaluate_10x():
    """Evaluate 10x model performance."""
    
    print("="*60)
    print("EVALUATING 10X MODEL PERFORMANCE")
    print("="*60)
    
    model_path = 'checkpoints/10x/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run train_10x.py or train_10x_fast.py first")
        return
    
    # Test on different job sizes
    test_cases = [
        ('Small (40 jobs)', 'data/40_jobs.json'),
        ('Medium (60 jobs)', 'data/60_jobs.json'),
        ('Large (80 jobs)', 'data/80_jobs.json'),
        ('Extra Large (100 jobs)', 'data/100_jobs.json'),
    ]
    
    results = []
    
    for name, data_path in test_cases:
        if not os.path.exists(data_path):
            print(f"\n{name}: Data file not found")
            continue
            
        print(f"\nTesting on {name}")
        print("-" * 40)
        
        # Create environment
        env = SchedulingEnv(data_path, max_steps=5000)
        
        # Load 10x model
        model = PPOScheduler(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_sizes=(512, 512, 256, 128),
            dropout_rate=0.1,
            use_batch_norm=False,
            exploration_rate=0,  # No exploration
            device='mps'
        )
        
        try:
            model.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model may need more training")
            return
        
        # Disable exploration
        if hasattr(model, 'set_training_mode'):
            model.set_training_mode(False)
        model.exploration_rate = 0
        
        # Run scheduling
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 5000:
            action, _ = model.predict(obs, info['action_mask'], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Calculate metrics
        completion_rate = info['tasks_scheduled'] / info['total_tasks']
        
        # Find makespan
        makespan = 0
        for task_idx, (start, end, machine) in env.task_schedules.items():
            makespan = max(makespan, end)
        
        # Count late jobs
        late_jobs = 0
        on_time_jobs = 0
        for task_idx, (start, end, machine) in env.task_schedules.items():
            task = env.loader.tasks[task_idx]
            family = env.loader.families[task.family_id]
            lcd_hours = family.lcd_days_remaining * 24
            if end > lcd_hours:
                late_jobs += 1
            else:
                on_time_jobs += 1
        
        late_percentage = late_jobs / info['total_tasks'] if info['total_tasks'] > 0 else 0
        
        # Machine utilization
        total_processing = sum(t.processing_time for t in env.loader.tasks)
        n_machines = len(env.loader.machines)
        theoretical_min = total_processing / n_machines
        efficiency = (theoretical_min / makespan * 100) if makespan > 0 else 0
        
        result = {
            'name': name,
            'completion_rate': completion_rate,
            'tasks_scheduled': info['tasks_scheduled'],
            'total_tasks': info['total_tasks'],
            'makespan': makespan,
            'late_jobs': late_jobs,
            'on_time_jobs': on_time_jobs,
            'late_percentage': late_percentage,
            'efficiency': efficiency,
            'reward': reward,
            'steps': steps
        }
        results.append(result)
        
        print(f"  Completion: {completion_rate:.1%} ({info['tasks_scheduled']}/{info['total_tasks']})")
        print(f"  Makespan: {makespan:.1f} hours ({makespan/24:.1f} days)")
        print(f"  On-time: {on_time_jobs}/{info['total_tasks']} ({(1-late_percentage):.1%})")
        print(f"  Late: {late_jobs}/{info['total_tasks']} ({late_percentage:.1%})")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Reward: {reward:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if results:
        avg_completion = sum(r['completion_rate'] for r in results) / len(results)
        avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
        avg_on_time = sum(1 - r['late_percentage'] for r in results) / len(results)
        
        print(f"Average Completion Rate: {avg_completion:.1%}")
        print(f"Average On-Time Rate: {avg_on_time:.1%}")
        print(f"Average Efficiency: {avg_efficiency:.1f}%")
        
        # Performance assessment
        print("\n" + "="*60)
        print("MODEL ASSESSMENT")
        print("="*60)
        
        if avg_completion >= 0.99:
            print("EXCELLENT: Model achieves near-perfect task completion!")
        elif avg_completion >= 0.95:
            print("GOOD: Model performs well with high completion rates")
        else:
            print("NEEDS IMPROVEMENT: Model requires more training")
        
        if avg_efficiency > 70:
            print("EFFICIENT: Good machine utilization and makespan")
        elif avg_efficiency > 50:
            print("MODERATE: Reasonable efficiency, room for improvement")
        else:
            print("INEFFICIENT: Poor machine utilization")
        
        # Model size info
        print("\n" + "="*60)
        print("MODEL SPECIFICATIONS")
        print("="*60)
        print(f"Architecture: 512→512→256→128 (4 layers)")
        print(f"Parameters: ~1.1M (4x larger than original)")
        print(f"Features: Dropout, LayerNorm, Smart Exploration")
        print(f"Training: Enhanced rewards, curriculum learning")
    
    print("="*60)

if __name__ == "__main__":
    evaluate_10x()