"""
Test final results - can it schedule ALL jobs?
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from phase3.environments.schedule_all_env import ScheduleAllEnvironment


def test_model(model_path, stage_name):
    """Test if model can schedule all jobs."""
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = ScheduleAllEnvironment(stage_name, verbose=False)
    
    # Test for 5 episodes
    results = []
    
    print(f"\nTesting {stage_name}:")
    print("-" * 40)
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            steps += 1
        
        scheduled = len(env.scheduled_jobs)
        total = env.total_tasks
        rate = scheduled / total
        results.append(rate)
        
        status = "✓ ALL JOBS SCHEDULED!" if rate == 1.0 else f"✗ Only {rate:.1%}"
        print(f"Episode {ep+1}: {scheduled}/{total} jobs in {steps} steps - {status}")
    
    avg_rate = np.mean(results)
    print(f"\nAverage: {avg_rate:.1%}")
    
    if avg_rate >= 0.99:
        print(f"✓ {stage_name.upper()} CAN SCHEDULE ALL JOBS!")
    elif avg_rate >= 0.8:
        print(f"✓ {stage_name.upper()} ACHIEVED 80% TARGET!")
    else:
        print(f"✗ {stage_name.upper()} needs more training")
    
    return avg_rate


def main():
    """Test all models."""
    print("FINAL RESULTS - CAN IT SCHEDULE ALL JOBS?")
    print("=" * 60)
    
    models = [
        ("toy_normal", "/Users/carrickcheah/Project/ppo/app_2/phase3/models_schedule_all/toy_normal_schedule_all.zip"),
        ("toy_hard", "/Users/carrickcheah/Project/ppo/app_2/phase3/models_schedule_all/toy_hard_schedule_all.zip"),
        ("toy_multi", "/Users/carrickcheah/Project/ppo/app_2/phase3/models_schedule_all/toy_multi_schedule_all.zip")
    ]
    
    summary = {}
    
    for stage_name, model_path in models:
        rate = test_model(model_path, stage_name)
        summary[stage_name] = rate
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    baseline = {'toy_normal': 0.562, 'toy_hard': 0.30, 'toy_multi': 0.364}
    
    for stage in ['toy_normal', 'toy_hard', 'toy_multi']:
        old = baseline[stage]
        new = summary[stage]
        improvement = new - old
        
        print(f"{stage:12} {old:.1%} → {new:.1%} (improved {improvement:+.1%})")
    
    all_pass = all(rate >= 0.8 for rate in summary.values())
    if all_pass:
        print("\n✓ ALL STAGES ACHIEVED 80%+ TARGET!")
        if all(rate >= 0.99 for rate in summary.values()):
            print("✓ ALL STAGES CAN SCHEDULE ALL JOBS!")
    else:
        print("\n✗ Some stages still need improvement")


if __name__ == "__main__":
    main()