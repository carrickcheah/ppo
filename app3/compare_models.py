#!/usr/bin/env python
"""
Compare models before and after training to see if improved or got worse
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

def evaluate_model(model_path, model_name, env, architecture_params=None):
    """Evaluate a single model's performance."""
    
    print(f"\nEvaluating: {model_name}")
    print("-" * 40)
    
    # Create model with appropriate architecture
    if architecture_params:
        model = PPOScheduler(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **architecture_params,
            device='mps'
        )
    else:
        # Default small architecture
        model = PPOScheduler(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='mps'
        )
    
    # Load model weights
    if os.path.exists(model_path):
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='mps', weights_only=False)
            model_dict = model.policy.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['policy_state_dict'].items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.policy.load_state_dict(model_dict, strict=False)
        except:
            print(f"  Warning: Could not fully load {model_name}")
    else:
        print(f"  Model not found: {model_path}")
        return None
    
    # Disable exploration
    if hasattr(model, 'set_training_mode'):
        model.set_training_mode(False)
    model.exploration_rate = 0
    
    # Run scheduling
    start_time = time.time()
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 10000:
        action, _ = model.predict(obs, info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    scheduling_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {}
    
    # 1. Completion rate
    metrics['completion_rate'] = info['tasks_scheduled'] / info['total_tasks']
    metrics['tasks_scheduled'] = info['tasks_scheduled']
    metrics['total_tasks'] = info['total_tasks']
    
    # 2. Makespan
    if env.task_schedules:
        metrics['makespan'] = max(end for _, end, _ in env.task_schedules.values())
    else:
        metrics['makespan'] = 0
    
    # 3. Machine utilization
    total_processing = sum(t.processing_time for t in env.loader.tasks[:metrics['tasks_scheduled']])
    n_machines = len(env.loader.machines)
    if metrics['makespan'] > 0:
        metrics['utilization'] = (total_processing / (metrics['makespan'] * n_machines)) * 100
    else:
        metrics['utilization'] = 0
    
    # 4. On-time delivery
    late_jobs = 0
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        family = env.loader.families[task.family_id]
        lcd_hours = family.lcd_days_remaining * 24
        if end > lcd_hours:
            late_jobs += 1
    
    metrics['late_jobs'] = late_jobs
    metrics['on_time_rate'] = 1 - (late_jobs / metrics['tasks_scheduled']) if metrics['tasks_scheduled'] > 0 else 0
    
    # 5. Performance
    metrics['scheduling_time'] = scheduling_time
    metrics['steps'] = steps
    metrics['final_reward'] = float(reward)
    
    # 6. Sequence violations
    sequence_violations = 0
    family_tasks = {}
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        if task.family_id not in family_tasks:
            family_tasks[task.family_id] = []
        family_tasks[task.family_id].append({
            'sequence': task.sequence,
            'start': start,
            'end': end
        })
    
    for family_id, tasks in family_tasks.items():
        tasks.sort(key=lambda x: x['sequence'])
        for i in range(len(tasks) - 1):
            if tasks[i]['end'] > tasks[i+1]['start']:
                sequence_violations += 1
    
    metrics['sequence_violations'] = sequence_violations
    
    return metrics

def compare_models():
    """Compare different model versions to track improvement."""
    
    print("="*80)
    print("MODEL COMPARISON - IS IT GETTING BETTER OR WORSE?")
    print("="*80)
    
    # Define models to compare
    models_to_compare = [
        {
            'path': 'checkpoints/fast/best_model.pth',
            'name': 'Original Model',
            'architecture': {
                'hidden_sizes': (256, 128, 64),
                'dropout_rate': 0,
                'use_batch_norm': False,
                'exploration_rate': 0
            }
        },
        {
            'path': 'checkpoints/10x/checkpoint_100.pth',
            'name': 'After 100 Episodes',
            'architecture': {
                'hidden_sizes': (512, 512, 256, 128),
                'dropout_rate': 0.1,
                'use_batch_norm': False,
                'exploration_rate': 0
            }
        },
        {
            'path': 'checkpoints/10x/checkpoint_500.pth',
            'name': 'After 500 Episodes',
            'architecture': {
                'hidden_sizes': (512, 512, 256, 128),
                'dropout_rate': 0.1,
                'use_batch_norm': False,
                'exploration_rate': 0
            }
        },
        {
            'path': 'checkpoints/10x/best_model.pth',
            'name': 'Best 10x Model',
            'architecture': {
                'hidden_sizes': (512, 512, 256, 128),
                'dropout_rate': 0.1,
                'use_batch_norm': False,
                'exploration_rate': 0
            }
        }
    ]
    
    # Test on same environment for fair comparison
    print("\nLoading test environment...")
    env = SchedulingEnv('data/40_jobs.json', max_steps=5000)
    
    # Evaluate each model
    results = []
    for model_config in models_to_compare:
        if os.path.exists(model_config['path']):
            metrics = evaluate_model(
                model_config['path'],
                model_config['name'],
                env,
                model_config['architecture']
            )
            if metrics:
                metrics['name'] = model_config['name']
                results.append(metrics)
    
    if len(results) < 2:
        print("\n❌ Not enough models to compare. Train more checkpoints first.")
        return
    
    # Display comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    # Header
    print(f"\n{'Metric':<25} ", end="")
    for r in results:
        print(f"{r['name']:<20} ", end="")
    print("\n" + "-"*85)
    
    # Key metrics to compare
    metrics_to_show = [
        ('Completion Rate (%)', 'completion_rate', lambda x: f"{x*100:.1f}%", True),
        ('Tasks Scheduled', 'tasks_scheduled', lambda x: f"{x}/{results[0]['total_tasks']}", True),
        ('Makespan (hours)', 'makespan', lambda x: f"{x:.1f}", False),
        ('Utilization (%)', 'utilization', lambda x: f"{x:.1f}%", True),
        ('On-Time Rate (%)', 'on_time_rate', lambda x: f"{x*100:.1f}%", True),
        ('Late Jobs', 'late_jobs', lambda x: f"{x}", False),
        ('Sequence Violations', 'sequence_violations', lambda x: f"{x}", False),
        ('Final Reward', 'final_reward', lambda x: f"{x:.1f}", True),
        ('Scheduling Time (s)', 'scheduling_time', lambda x: f"{x:.2f}", False),
    ]
    
    # Print each metric
    for display_name, key, formatter, higher_better in metrics_to_show:
        print(f"{display_name:<25} ", end="")
        
        values = []
        for r in results:
            value = r.get(key, 0)
            values.append(value)
            print(f"{formatter(value):<20} ", end="")
        
        # Show trend
        if len(values) >= 2:
            if higher_better:
                if values[-1] > values[0]:
                    print("↑ ✅", end="")
                elif values[-1] < values[0]:
                    print("↓ ❌", end="")
                else:
                    print("→", end="")
            else:  # Lower is better
                if values[-1] < values[0]:
                    print("↓ ✅", end="")
                elif values[-1] > values[0]:
                    print("↑ ❌", end="")
                else:
                    print("→", end="")
        print()
    
    # Calculate improvement scores
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    if len(results) >= 2:
        original = results[0]
        latest = results[-1]
        
        improvements = []
        degradations = []
        
        # Check each metric
        checks = [
            ('Completion', 'completion_rate', True, 0.01),
            ('Makespan', 'makespan', False, 10),
            ('Utilization', 'utilization', True, 1),
            ('On-Time', 'on_time_rate', True, 0.05),
            ('Violations', 'sequence_violations', False, 1),
            ('Reward', 'final_reward', True, 10),
        ]
        
        for name, key, higher_better, threshold in checks:
            orig_val = original.get(key, 0)
            new_val = latest.get(key, 0)
            
            if higher_better:
                diff = new_val - orig_val
                if diff > threshold:
                    improvements.append(f"{name} (+{diff:.1f})")
                elif diff < -threshold:
                    degradations.append(f"{name} ({diff:.1f})")
            else:
                diff = orig_val - new_val
                if diff > threshold:
                    improvements.append(f"{name} (-{abs(new_val-orig_val):.1f})")
                elif diff < -threshold:
                    degradations.append(f"{name} (+{abs(new_val-orig_val):.1f})")
        
        print(f"\n📊 Comparing '{original['name']}' → '{latest['name']}':")
        
        if improvements:
            print(f"\n✅ IMPROVEMENTS:")
            for imp in improvements:
                print(f"   • {imp}")
        
        if degradations:
            print(f"\n❌ DEGRADATIONS:")
            for deg in degradations:
                print(f"   • {deg}")
        
        # Overall verdict
        print("\n" + "="*80)
        print("VERDICT: IS THE MODEL BETTER?")
        print("="*80)
        
        # Score calculation
        score_original = (
            original['completion_rate'] * 40 +
            original['on_time_rate'] * 30 +
            (original['utilization'] / 100) * 20 +
            (1 - original['sequence_violations'] / 10) * 10
        )
        
        score_latest = (
            latest['completion_rate'] * 40 +
            latest['on_time_rate'] * 30 +
            (latest['utilization'] / 100) * 20 +
            (1 - latest['sequence_violations'] / 10) * 10
        )
        
        improvement_percentage = ((score_latest - score_original) / score_original * 100) if score_original > 0 else 0
        
        print(f"\nOriginal Score: {score_original:.1f}/100")
        print(f"Latest Score: {score_latest:.1f}/100")
        print(f"Change: {improvement_percentage:+.1f}%")
        
        if improvement_percentage > 10:
            print("\n🎉 MODEL IS SIGNIFICANTLY BETTER!")
            print("Training is working well. Continue training.")
        elif improvement_percentage > 0:
            print("\n✅ MODEL IS SLIGHTLY BETTER")
            print("Some improvement, but could be better.")
        elif improvement_percentage > -5:
            print("\n➡️ MODEL IS ABOUT THE SAME")
            print("No significant change. May need more training or tuning.")
        else:
            print("\n⚠️ MODEL GOT WORSE!")
            print("Training may be overfitting or hyperparameters need adjustment.")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"model_comparison_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Create visualization if we have multiple checkpoints
    if len(results) > 2:
        create_training_curve(results)

def create_training_curve(results):
    """Create visualization of training progress."""
    
    print("\nCreating training progress visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Training Progress', fontsize=16, fontweight='bold')
    
    metrics = [
        ('Completion Rate', 'completion_rate', axes[0, 0], True),
        ('On-Time Rate', 'on_time_rate', axes[0, 1], True),
        ('Utilization', 'utilization', axes[0, 2], True),
        ('Makespan', 'makespan', axes[1, 0], False),
        ('Sequence Violations', 'sequence_violations', axes[1, 1], False),
        ('Final Reward', 'final_reward', axes[1, 2], True),
    ]
    
    x_labels = [r['name'].replace(' Model', '').replace('After ', '') for r in results]
    x_pos = range(len(results))
    
    for title, key, ax, higher_better in metrics:
        values = [r.get(key, 0) for r in results]
        
        # Determine color based on trend
        if len(values) >= 2:
            if higher_better:
                color = 'green' if values[-1] > values[0] else 'red'
            else:
                color = 'green' if values[-1] < values[0] else 'red'
        else:
            color = 'blue'
        
        ax.plot(x_pos, values, marker='o', linewidth=2, markersize=8, color=color)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            if key in ['completion_rate', 'on_time_rate']:
                label = f"{v*100:.1f}%"
            elif key == 'utilization':
                label = f"{v:.1f}%"
            else:
                label = f"{v:.1f}"
            ax.text(i, v, label, ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_file = 'training_progress.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    compare_models()