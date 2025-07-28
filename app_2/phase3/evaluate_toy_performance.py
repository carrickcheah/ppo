"""
Evaluate toy stage models performance
Check if they achieve 100% scheduling
"""

import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

def evaluate_model(stage_name, model_path=None):
    """Evaluate a toy model's performance"""
    
    print(f"\nEvaluating {stage_name}:")
    print("-" * 50)
    
    # Default model paths to check
    if model_path is None:
        model_paths = [
            f"/Users/carrickcheah/Project/ppo/app_2/phase3/phased_models/{stage_name}_phased.zip",
            f"/Users/carrickcheah/Project/ppo/app_2/phase3/perfect_models/{stage_name}_100percent.zip",
            f"/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models/{stage_name}_final.zip",
            f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/perfect/{stage_name}/{stage_name}_checkpoint_100000_steps.zip"
        ]
        
        # Find first existing model
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path or not os.path.exists(model_path):
        print(f"No model found for {stage_name}")
        return None
    
    print(f"Using model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    # Run multiple episodes for robust evaluation
    n_episodes = 20
    results = {
        'scheduled_jobs': [],
        'total_jobs': [],
        'rewards': [],
        'steps': [],
        'completion_rates': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = done or truncated
        
        scheduled = len(env.scheduled_jobs) if hasattr(env, 'scheduled_jobs') else 0
        total = env.total_tasks if hasattr(env, 'total_tasks') else 1
        completion_rate = scheduled / total if total > 0 else 0
        
        results['scheduled_jobs'].append(scheduled)
        results['total_jobs'].append(total)
        results['rewards'].append(ep_reward)
        results['steps'].append(steps)
        results['completion_rates'].append(completion_rate)
        
        if ep == 0:
            print(f"  First episode: {scheduled}/{total} jobs ({completion_rate:.1%}), reward: {ep_reward:.1f}")
    
    # Calculate statistics
    avg_completion = np.mean(results['completion_rates'])
    std_completion = np.std(results['completion_rates'])
    avg_reward = np.mean(results['rewards'])
    min_completion = np.min(results['completion_rates'])
    max_completion = np.max(results['completion_rates'])
    
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Average completion: {avg_completion:.1%} (±{std_completion:.1%})")
    print(f"  Min/Max completion: {min_completion:.1%} / {max_completion:.1%}")
    print(f"  Average reward: {avg_reward:.1f}")
    
    return {
        'stage': stage_name,
        'model_path': model_path,
        'avg_completion': avg_completion,
        'std_completion': std_completion,
        'avg_reward': avg_reward,
        'min_completion': min_completion,
        'max_completion': max_completion,
        'is_perfect': avg_completion >= 0.99
    }

def main():
    """Evaluate all toy stages"""
    
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Evaluating Toy Stage Performance")
    print("=" * 60)
    print("Target: 100% scheduling for all stages")
    
    all_results = {}
    
    for stage in stages:
        result = evaluate_model(stage)
        if result:
            all_results[stage] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Stage':<15} {'Avg Completion':<20} {'Consistency':<15} {'Status':<20}")
    print("-" * 70)
    
    all_perfect = True
    
    for stage in stages:
        if stage in all_results:
            r = all_results[stage]
            avg = f"{r['avg_completion']:.1%}"
            consistency = f"±{r['std_completion']:.1%}"
            
            if r['is_perfect']:
                status = "✓ PERFECT!"
            elif r['avg_completion'] >= 0.9:
                status = "✓ Good (>90%)"
                all_perfect = False
            elif r['avg_completion'] >= 0.7:
                status = "⚠ Needs improvement"
                all_perfect = False
            else:
                status = "✗ Poor performance"
                all_perfect = False
        else:
            avg = "N/A"
            consistency = "N/A"
            status = "✗ No model found"
            all_perfect = False
        
        print(f"{stage:<15} {avg:<20} {consistency:<15} {status:<20}")
    
    print("\n" + "=" * 60)
    
    if all_perfect:
        print("✓ ALL TOY STAGES ACHIEVED 100% PERFORMANCE!")
        print("Ready to proceed to the next phase!")
    else:
        print("✗ Some stages still need improvement")
        print("\nNext steps:")
        for stage, r in all_results.items():
            if not r['is_perfect']:
                print(f"  - {stage}: Train longer or adjust rewards (current: {r['avg_completion']:.1%})")

if __name__ == "__main__":
    main()