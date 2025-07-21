#!/usr/bin/env python3
"""
Evaluate the extended model performance to verify <45h makespan target
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv
import json
from datetime import datetime

print("="*60)
print("PHASE 4 EXTENDED MODEL EVALUATION")
print("="*60)
print("Target: <45h makespan\n")

# Model paths
original_model_path = "models/full_production/final_model.zip"
extended_model_path = "models/full_production/extended/final_extended_model.zip"

# Evaluate both models
results = {}

for model_name, model_path in [("Original (1M steps)", original_model_path), 
                                ("Extended (1.5M steps)", extended_model_path)]:
    print(f"\nEvaluating {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"  Model not found at {model_path}")
        continue
        
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = FullProductionEnv(
        n_machines=150,
        n_jobs=500,
        state_compression="hierarchical",
        use_break_constraints=True,
        use_holiday_constraints=True,
        seed=42
    )
    
    # Run evaluation episodes
    episode_results = []
    n_episodes = 5
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
        # Collect episode stats
        makespan = info.get('makespan', 0)
        utilization = info.get('avg_utilization', 0)
        completion = info.get('completion_rate', 1.0)
        
        episode_results.append({
            'episode': episode + 1,
            'makespan': makespan,
            'utilization': utilization,
            'completion': completion,
            'reward': episode_reward,
            'steps': steps
        })
        
        print(f"  Episode {episode + 1}: {makespan:.1f}h makespan")
    
    env.close()
    
    # Calculate averages
    avg_makespan = np.mean([r['makespan'] for r in episode_results])
    avg_utilization = np.mean([r['utilization'] for r in episode_results])
    avg_completion = np.mean([r['completion'] for r in episode_results])
    
    results[model_name] = {
        'avg_makespan': avg_makespan,
        'avg_utilization': avg_utilization,
        'avg_completion': avg_completion,
        'episodes': episode_results
    }
    
    print(f"\n  Average Results:")
    print(f"    Makespan: {avg_makespan:.2f}h")
    print(f"    Completion: {avg_completion:.1%}")
    print(f"    Utilization: {avg_utilization:.1%}")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

if len(results) == 2:
    original = results["Original (1M steps)"]
    extended = results["Extended (1.5M steps)"]
    
    makespan_reduction = original['avg_makespan'] - extended['avg_makespan']
    makespan_improvement = (makespan_reduction / original['avg_makespan']) * 100
    
    print(f"Original Model: {original['avg_makespan']:.2f}h")
    print(f"Extended Model: {extended['avg_makespan']:.2f}h")
    print(f"Improvement: {makespan_reduction:.2f}h ({makespan_improvement:.1f}%)")
    
    if extended['avg_makespan'] < 45:
        print(f"\n✅ SUCCESS! Target <45h achieved: {extended['avg_makespan']:.2f}h")
    else:
        print(f"\n❌ Target not met. Current: {extended['avg_makespan']:.2f}h, Target: <45h")
        print(f"   Gap: {extended['avg_makespan'] - 45:.2f}h")

# Save results
results_data = {
    'evaluation_date': datetime.now().isoformat(),
    'target_makespan': 45.0,
    'results': results
}

with open('phase_4/extended_model_evaluation.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\nResults saved to extended_model_evaluation.json")