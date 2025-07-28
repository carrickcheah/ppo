"""
Final evaluation of all trained models
"""

import os
import sys
import json
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

print("="*60)
print("FINAL EVALUATION OF TRAINED MODELS")
print("="*60)

# Model locations to check
model_locations = {
    'toy_easy': [
        "/Users/carrickcheah/Project/ppo/app_2/phase3/final_models/toy_easy_model.zip",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/fixed_models/toy_easy_fixed_model.zip",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/toy_easy/final_model.zip"
    ],
    'toy_normal': [
        "/Users/carrickcheah/Project/ppo/app_2/phase3/final_models/toy_normal_model.zip",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/toy_normal/final_model.zip"
    ],
    'toy_hard': [
        "/Users/carrickcheah/Project/ppo/app_2/phase3/final_models/toy_hard_model.zip",
        "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/toy_hard/final_model.zip"
    ],
    'toy_multi': [
        "/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/foundation/toy_multi/final_model.zip"
    ]
}

results = {}

for stage, paths in model_locations.items():
    print(f"\nEvaluating {stage}...")
    print("-"*40)
    
    # Find available model
    model_path = None
    for path in paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"  No model found for {stage}")
        continue
    
    # Load model
    model = PPO.load(model_path)
    env = CurriculumEnvironmentTrulyFixed(stage, verbose=False)
    
    # Evaluate
    eval_episodes = 20
    total_scheduled = 0
    total_possible = 0
    episode_rewards = []
    
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        actions_taken = {'schedule': 0, 'no_action': 0, 'invalid': 0}
        
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            
            action_type = info.get('action_type', 'unknown')
            if action_type in actions_taken:
                actions_taken[action_type] += 1
            
            steps += 1
        
        scheduled = len(env.scheduled_jobs)
        total = env.total_tasks
        total_scheduled += scheduled
        total_possible += total
        episode_rewards.append(ep_reward)
    
    # Calculate metrics
    scheduling_rate = total_scheduled / total_possible if total_possible > 0 else 0
    avg_reward = np.mean(episode_rewards)
    
    results[stage] = {
        'scheduling_rate': scheduling_rate,
        'avg_reward': avg_reward,
        'model_path': model_path
    }
    
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Scheduling rate: {scheduling_rate:.1%}")
    print(f"  Average reward: {avg_reward:.1f}")
    
    if scheduling_rate >= 0.8:
        print(f"  Status: EXCELLENT!")
    elif scheduling_rate >= 0.5:
        print(f"  Status: Good progress")
    else:
        print(f"  Status: Needs improvement")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/results/q_final_evaluation_{timestamp}.json"

summary = {
    'evaluation_date': datetime.now().isoformat(),
    'stages': results,
    'overall_metrics': {
        'avg_scheduling_rate': np.mean([r['scheduling_rate'] for r in results.values()]) if results else 0,
        'stages_evaluated': len(results),
        'stages_above_50pct': sum(1 for r in results.values() if r['scheduling_rate'] > 0.5)
    }
}

with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nStages evaluated: {len(results)}/4")
print(f"Average scheduling rate: {summary['overall_metrics']['avg_scheduling_rate']:.1%}")
print(f"Stages above 50%: {summary['overall_metrics']['stages_above_50pct']}")
print(f"\nResults saved to: {output_path}")

if summary['overall_metrics']['avg_scheduling_rate'] > 0.5:
    print("\n✓ OVERALL: Good progress! Models are learning to schedule.")
else:
    print("\n× OVERALL: More training needed.")