#!/usr/bin/env python
"""
Test the 1 million step model and compare with ALL previous models
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from src.environments.scheduling_env import SchedulingEnv
import numpy as np
import time

def test_1million_model():
    """Test 1M model and provide comprehensive comparison."""
    
    print("="*80)
    print("🎯 TESTING 1 MILLION STEP MODEL")
    print("="*80)
    
    # Find the model
    model_paths = [
        "checkpoints/sb3_1million/final_1million_model.zip",
        "checkpoints/sb3_1million/best_model.zip",
        "checkpoints/sb3_1million/checkpoint_100000_steps.zip"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No 1M model found yet. Training may still be in progress.")
        print("Check: tail -f training_1m_fixed.log")
        return None
    
    print(f"Loading model: {model_path}")
    file_size = os.path.getsize(model_path) / (1024*1024)
    print(f"Model size: {file_size:.1f} MB")
    
    model = PPO.load(model_path)
    
    # Test on 100 jobs
    data_path = 'data/100_jobs.json'
    
    print(f"\n📊 Testing on: {data_path}")
    print("-"*60)
    
    # Run 3 tests for reliability
    results = []
    
    for run in range(3):
        print(f"\nRun {run+1}/3:")
        
        env = SchedulingEnv(data_path, max_steps=10000)
        obs, info = env.reset(seed=run*100)
        
        start_time = time.time()
        done = False
        steps = 0
        
        while not done and steps < 10000:
            action, _ = model.predict(obs, deterministic=True)
            
            if 'action_mask' in info:
                mask = info['action_mask']
                if not mask[action]:
                    valid_actions = np.where(mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
            
            if steps % 2000 == 0:
                print(f"  Step {steps}: {info['tasks_scheduled']}/{info['total_tasks']} scheduled")
        
        test_time = time.time() - start_time
        
        # Calculate metrics
        schedule = env.get_final_schedule()
        if schedule['tasks']:
            total_processing = sum(t['processing_time'] for t in schedule['tasks'])
            makespan = max(t['end'] for t in schedule['tasks'])
            n_machines = len(env.loader.machines)
            efficiency = (total_processing / n_machines / makespan * 100)
            
            late_jobs = sum(1 for t in schedule['tasks'] if t['end'] > t['lcd_days'] * 24)
            on_time_rate = (1 - late_jobs / len(schedule['tasks'])) * 100
        else:
            efficiency = 0
            on_time_rate = 0
        
        completion_rate = info['tasks_scheduled'] / info['total_tasks'] * 100
        
        results.append({
            'completion': completion_rate,
            'efficiency': efficiency,
            'on_time': on_time_rate,
            'reward': env.episode_reward,
            'makespan': makespan if schedule['tasks'] else 0,
            'steps': steps,
            'time': test_time
        })
        
        print(f"  Completion: {completion_rate:.1f}%")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Reward: {env.episode_reward:.0f}")
    
    # Average results
    avg_completion = np.mean([r['completion'] for r in results])
    avg_efficiency = np.mean([r['efficiency'] for r in results])
    avg_on_time = np.mean([r['on_time'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"\n{'='*80}")
    print("📊 AVERAGE PERFORMANCE (3 runs)")
    print(f"{'='*80}")
    
    print(f"\n1 Million Step Model Results:")
    print(f"  - Completion: {avg_completion:.1f}%")
    print(f"  - Efficiency: {avg_efficiency:.1f}%")
    print(f"  - On-time: {avg_on_time:.1f}%")
    print(f"  - Reward: {avg_reward:.0f}")
    print(f"  - Steps: {avg_steps:.0f}")
    
    # COMPREHENSIVE COMPARISON
    print(f"\n{'='*80}")
    print("📊 FULL COMPARISON - ALL MODELS")
    print(f"{'='*80}")
    
    all_models = {
        'Custom PPO (baseline)': {'efficiency': 7.4, 'completion': 100},
        'SB3 50k (demo)': {'efficiency': 3.1, 'completion': 100},
        'SB3 100k (stage 1)': {'efficiency': 3.1, 'completion': 100},
        'SB3 25k (optimized)': {'efficiency': 8.9, 'completion': 100},
        'SB3 1M (current)': {'efficiency': avg_efficiency, 'completion': avg_completion}
    }
    
    print(f"\n{'Model':<25} {'Efficiency':>12} {'Completion':>12} {'vs Baseline':>15} {'Achievement':>20}")
    print("-"*90)
    
    baseline_eff = 7.4
    for name, metrics in all_models.items():
        eff = metrics['efficiency']
        comp = metrics['completion']
        improvement = eff / baseline_eff
        
        if improvement >= 100:
            achievement = "🏆🏆🏆 100x!"
        elif improvement >= 50:
            achievement = "🏆🏆 50x!"
        elif improvement >= 20:
            achievement = "🏆 20x!"
        elif improvement >= 10:
            achievement = "✅ 10x!"
        elif improvement >= 5:
            achievement = "✅ 5x!"
        elif improvement >= 2:
            achievement = "📈 2x"
        elif improvement >= 1:
            achievement = "📈 >1x"
        else:
            achievement = ""
        
        print(f"{name:<25} {eff:>11.1f}% {comp:>11.1f}% {improvement:>14.1f}x {achievement:>20}")
    
    improvement = avg_efficiency / baseline_eff
    
    # DETAILED ANALYSIS
    print(f"\n{'='*80}")
    print("📈 DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    print("\n🔍 Performance Breakdown:")
    print(f"  - Base efficiency (custom): {baseline_eff:.1f}%")
    print(f"  - Current efficiency (1M): {avg_efficiency:.1f}%")
    print(f"  - Absolute gain: {avg_efficiency - baseline_eff:+.1f}%")
    print(f"  - Relative improvement: {improvement:.1f}x")
    
    print("\n📊 Training Progression:")
    print("  Steps     | Efficiency | Improvement")
    print("  ----------|------------|------------")
    print(f"  50k       | 3.1%       | 0.4x")
    print(f"  100k      | 3.1%       | 0.4x")
    print(f"  25k (opt) | 8.9%       | 1.2x")
    print(f"  1M        | {avg_efficiency:.1f}%      | {improvement:.1f}x")
    
    # SUCCESS CRITERIA
    print(f"\n{'='*80}")
    print("🎯 SUCCESS CRITERIA")
    print(f"{'='*80}")
    
    criteria = [
        ("Task Completion", avg_completion >= 99, f"{avg_completion:.1f}%"),
        ("10x Improvement", improvement >= 10, f"{improvement:.1f}x"),
        ("20x Improvement", improvement >= 20, f"{improvement:.1f}x"),
        ("50x Improvement", improvement >= 50, f"{improvement:.1f}x"),
        ("100x Improvement", improvement >= 100, f"{improvement:.1f}x"),
    ]
    
    print(f"\n{'Criterion':<20} {'Status':>10} {'Value':>15}")
    print("-"*45)
    for criterion, met, value in criteria:
        status = "✅ PASS" if met else "❌ FAIL"
        print(f"{criterion:<20} {status:>10} {value:>15}")
    
    # FINAL VERDICT
    print(f"\n{'='*80}")
    print("🏁 FINAL VERDICT")
    print(f"{'='*80}")
    
    if improvement >= 100:
        print("\n🏆🏆🏆 100x IMPROVEMENT ACHIEVED! 🏆🏆🏆")
        print("CONGRATULATIONS! The goal has been reached!")
        print(f"Final improvement: {improvement:.1f}x")
    elif improvement >= 50:
        print("\n🏆🏆 50x IMPROVEMENT ACHIEVED!")
        print("Outstanding progress! Halfway to 100x goal.")
        print(f"Current: {improvement:.1f}x | Target: 100x")
    elif improvement >= 20:
        print("\n🏆 20x IMPROVEMENT ACHIEVED!")
        print("Excellent progress from 1M steps!")
        print(f"Current: {improvement:.1f}x | Target: 100x")
    elif improvement >= 10:
        print("\n✅ 10x IMPROVEMENT ACHIEVED!")
        print("Good milestone reached with 1M steps.")
        print(f"Current: {improvement:.1f}x | Target: 100x")
    else:
        print(f"\n📈 {improvement:.1f}x improvement achieved")
        print("More training or hyperparameter tuning needed.")
    
    # NEXT STEPS
    print(f"\n{'='*80}")
    print("📋 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if improvement < 100:
        steps_needed = int(1_000_000 * (100 / improvement))
        print(f"\n To reach 100x:")
        print(f"  - Current rate: {improvement:.1f}x per 1M steps")
        print(f"  - Estimated steps needed: {steps_needed:,}")
        print(f"  - Additional training: {(steps_needed - 1_000_000)/1_000_000:.1f}M steps")
        
        print("\n Optimization suggestions:")
        print("  1. Increase learning rate to 1e-3")
        print("  2. Use larger batch size (1024)")
        print("  3. Add more parallel environments (16)")
        print("  4. Tune reward weights further")
        print("  5. Use curriculum learning on harder problems")
    else:
        print("\n✅ Target achieved! Ready for production deployment.")
        print("  - Save the model for production use")
        print("  - Test on real-world scheduling scenarios")
        print("  - Compare with existing schedulers in production")
    
    print(f"{'='*80}")
    
    return avg_efficiency, improvement

if __name__ == "__main__":
    efficiency, improvement = test_1million_model()