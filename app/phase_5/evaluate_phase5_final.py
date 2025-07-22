#!/usr/bin/env python3
"""
Comprehensive evaluation of Phase 5 hierarchical model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

def evaluate_model(model_path: str, model_name: str):
    """Evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=1000,
        invalid_action_penalty=-5.0,
        seed=42
    )
    vec_env = DummyVecEnv([lambda: env])
    
    # Load model
    model = PPO.load(model_path)
    
    # Run full episode
    obs = vec_env.reset()
    
    metrics = {
        'scheduled_jobs': [],
        'invalid_actions': 0,
        'total_actions': 0,
        'rewards': [],
        'makespan': 0,
        'utilization': 0
    }
    
    done = False
    while not done and metrics['total_actions'] < 1000:
        # Use stochastic for better performance
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        
        metrics['total_actions'] += 1
        metrics['rewards'].append(reward[0])
        
        if info[0].get('invalid_action', False):
            metrics['invalid_actions'] += 1
        
        scheduled = info[0].get('scheduled_count', 0)
        metrics['scheduled_jobs'].append(scheduled)
        
        # Print progress every 100 steps
        if metrics['total_actions'] % 100 == 0:
            invalid_rate = metrics['invalid_actions'] / metrics['total_actions'] * 100
            print(f"Step {metrics['total_actions']}: {scheduled} jobs, {invalid_rate:.1f}% invalid")
        
        if done[0]:
            metrics['makespan'] = info[0].get('makespan', 0)
            metrics['utilization'] = info[0].get('avg_utilization', 0)
            break
    
    # Final statistics
    final_scheduled = metrics['scheduled_jobs'][-1] if metrics['scheduled_jobs'] else 0
    invalid_rate = metrics['invalid_actions'] / metrics['total_actions'] * 100
    avg_reward = np.mean(metrics['rewards']) if metrics['rewards'] else 0
    
    print(f"\nFinal Results:")
    print(f"  Jobs scheduled: {final_scheduled}/320 ({final_scheduled/320*100:.1f}%)")
    print(f"  Invalid action rate: {invalid_rate:.1f}%")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Total steps: {metrics['total_actions']}")
    
    if metrics['makespan'] > 0:
        print(f"  Makespan: {metrics['makespan']:.1f} hours")
        print(f"  Utilization: {metrics['utilization']:.1f}%")
    
    return metrics

def compare_models():
    """Compare different checkpoints"""
    models = [
        ("models/multidiscrete/exploration/phase5_explore_100000_steps", "100k steps"),
        ("models/multidiscrete/exploration/phase5_explore_300000_steps", "300k steps"),
        ("models/multidiscrete/correct_dims/phase5_320jobs_250000_steps", "250k correct dims"),
    ]
    
    # Check if 1M model exists
    if Path("models/multidiscrete/exploration_continued/model_1m.zip").exists():
        models.append(("models/multidiscrete/exploration_continued/model_1m", "1M steps"))
    
    results = {}
    for model_path, name in models:
        if Path(f"{model_path}.zip").exists():
            results[name] = evaluate_model(model_path, name)
        else:
            print(f"\nSkipping {name} - model not found")
    
    # Create comparison plot
    if len(results) > 1:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Jobs scheduled over time
        plt.subplot(2, 2, 1)
        for name, metrics in results.items():
            if metrics['scheduled_jobs']:
                plt.plot(metrics['scheduled_jobs'], label=name)
        plt.xlabel('Steps')
        plt.ylabel('Jobs Scheduled')
        plt.title('Scheduling Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Invalid action rates
        plt.subplot(2, 2, 2)
        names = list(results.keys())
        invalid_rates = [m['invalid_actions']/m['total_actions']*100 for m in results.values()]
        plt.bar(names, invalid_rates)
        plt.ylabel('Invalid Action Rate (%)')
        plt.title('Invalid Action Rates')
        plt.xticks(rotation=45)
        
        # Plot 3: Final jobs scheduled
        plt.subplot(2, 2, 3)
        final_jobs = [m['scheduled_jobs'][-1] if m['scheduled_jobs'] else 0 for m in results.values()]
        plt.bar(names, final_jobs)
        plt.axhline(y=320, color='r', linestyle='--', label='Target (320)')
        plt.ylabel('Jobs Scheduled')
        plt.title('Final Jobs Scheduled')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Plot 4: Average rewards
        plt.subplot(2, 2, 4)
        avg_rewards = [np.mean(m['rewards']) if m['rewards'] else 0 for m in results.values()]
        plt.bar(names, avg_rewards)
        plt.ylabel('Average Reward')
        plt.title('Average Rewards')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/phase_5/model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved to visualizations/phase_5/model_comparison.png")

def test_random_baseline():
    """Test random action baseline"""
    print("\n" + "="*60)
    print("Random Action Baseline")
    print("="*60 + "\n")
    
    env = MultiDiscreteHierarchicalEnv(
        n_machines=145,
        n_jobs=320,
        snapshot_file="data/real_production_snapshot.json",
        max_episode_steps=500,
        seed=999
    )
    vec_env = DummyVecEnv([lambda: env])
    
    obs = vec_env.reset()
    scheduled = 0
    invalid = 0
    
    for step in range(500):
        action = [vec_env.action_space.sample()]
        obs, reward, done, info = vec_env.step(action)
        
        if info[0].get('invalid_action', False):
            invalid += 1
        else:
            scheduled = info[0].get('scheduled_count', 0)
        
        if step % 100 == 0:
            print(f"Step {step}: {scheduled} jobs, {invalid/(step+1)*100:.1f}% invalid")
        
        if done[0]:
            break
    
    print(f"\nRandom baseline:")
    print(f"  Jobs scheduled: {scheduled}/320")
    print(f"  Invalid rate: {invalid/(step+1)*100:.1f}%")

if __name__ == "__main__":
    # Ensure visualization directory exists
    import os
    os.makedirs("visualizations/phase_5", exist_ok=True)
    
    # Test random baseline first
    test_random_baseline()
    
    # Compare all models
    compare_models()
    
    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)