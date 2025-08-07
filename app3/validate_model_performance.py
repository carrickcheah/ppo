#!/usr/bin/env python
"""
Comprehensive validation of PPO scheduling model performance
Checks if the model is actually good and results are valid
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments.scheduling_env import SchedulingEnv
from src.models.ppo_scheduler import PPOScheduler
import numpy as np
import time
from datetime import datetime

def validate_model_performance():
    """Validate if the model is good with comprehensive checks."""
    
    print("="*80)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*80)
    
    # Load environment and model
    data_path = 'data/100_jobs.json'
    model_path = 'checkpoints/10x/best_model.pth'
    
    env = SchedulingEnv(data_path, max_steps=10000)
    model = PPOScheduler(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=(512, 512, 256, 128),
        dropout_rate=0.1,
        use_batch_norm=False,
        exploration_rate=0,
        device='mps'
    )
    
    # Load model
    if os.path.exists(model_path):
        import torch
        checkpoint = torch.load(model_path, map_location='mps', weights_only=False)
        model_dict = model.policy.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['policy_state_dict'].items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.policy.load_state_dict(model_dict, strict=False)
    
    if hasattr(model, 'set_training_mode'):
        model.set_training_mode(False)
    model.exploration_rate = 0
    
    # Run scheduling
    print("\n1. Running PPO Scheduling...")
    print("-" * 40)
    
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
    
    # VALIDATION CRITERIA
    print("\n2. VALIDATION CHECKS")
    print("="*80)
    
    validation_results = []
    
    # ========================
    # CHECK 1: COMPLETION RATE
    # ========================
    print("\n✓ CHECK 1: Task Completion Rate")
    print("-" * 40)
    completion_rate = info['tasks_scheduled'] / info['total_tasks']
    print(f"  Tasks scheduled: {info['tasks_scheduled']}/{info['total_tasks']}")
    print(f"  Completion rate: {completion_rate:.1%}")
    
    if completion_rate >= 0.99:
        print("  ✅ EXCELLENT: Near perfect completion (>99%)")
        validation_results.append(("Completion", "PASS", 10))
    elif completion_rate >= 0.95:
        print("  ✅ GOOD: High completion (>95%)")
        validation_results.append(("Completion", "PASS", 8))
    elif completion_rate >= 0.90:
        print("  ⚠️  ACCEPTABLE: Moderate completion (>90%)")
        validation_results.append(("Completion", "WARN", 6))
    else:
        print("  ❌ FAIL: Low completion (<90%)")
        validation_results.append(("Completion", "FAIL", 0))
    
    # ========================
    # CHECK 2: SEQUENCE CONSTRAINTS
    # ========================
    print("\n✓ CHECK 2: Sequence Constraint Validation")
    print("-" * 40)
    
    sequence_violations = 0
    families_checked = 0
    
    # Group tasks by family
    family_tasks = {}
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        if task.family_id not in family_tasks:
            family_tasks[task.family_id] = []
        family_tasks[task.family_id].append({
            'task': task,
            'start': start,
            'end': end,
            'sequence': task.sequence
        })
    
    # Check sequence order within each family
    for family_id, tasks in family_tasks.items():
        families_checked += 1
        tasks.sort(key=lambda x: x['sequence'])
        
        for i in range(len(tasks) - 1):
            curr_task = tasks[i]
            next_task = tasks[i + 1]
            
            # Sequence i must complete before sequence i+1 starts
            if curr_task['sequence'] < next_task['sequence']:
                if curr_task['end'] > next_task['start']:
                    sequence_violations += 1
                    print(f"  ❌ Violation: Family {family_id}, "
                          f"Seq {curr_task['sequence']} ends at {curr_task['end']:.1f}, "
                          f"Seq {next_task['sequence']} starts at {next_task['start']:.1f}")
    
    print(f"  Families checked: {families_checked}")
    print(f"  Sequence violations: {sequence_violations}")
    
    if sequence_violations == 0:
        print("  ✅ PERFECT: No sequence violations")
        validation_results.append(("Sequences", "PASS", 10))
    elif sequence_violations <= 5:
        print("  ⚠️  MINOR: Few violations")
        validation_results.append(("Sequences", "WARN", 5))
    else:
        print("  ❌ FAIL: Many sequence violations")
        validation_results.append(("Sequences", "FAIL", 0))
    
    # ========================
    # CHECK 3: MACHINE CONFLICTS
    # ========================
    print("\n✓ CHECK 3: Machine Conflict Detection")
    print("-" * 40)
    
    machine_conflicts = 0
    machine_schedules = {}
    
    # Build machine schedules
    for task_idx, (start, end, machine) in env.task_schedules.items():
        if machine not in machine_schedules:
            machine_schedules[machine] = []
        machine_schedules[machine].append((start, end, task_idx))
    
    # Check for overlaps
    for machine, schedule in machine_schedules.items():
        schedule.sort(key=lambda x: x[0])  # Sort by start time
        
        for i in range(len(schedule) - 1):
            curr_end = schedule[i][1]
            next_start = schedule[i + 1][0]
            
            if curr_end > next_start + 0.01:  # Small tolerance for float comparison
                machine_conflicts += 1
                print(f"  ❌ Conflict on machine {machine}: "
                      f"Task {schedule[i][2]} ends at {curr_end:.1f}, "
                      f"Task {schedule[i+1][2]} starts at {next_start:.1f}")
    
    print(f"  Machines used: {len(machine_schedules)}/{len(env.loader.machines)}")
    print(f"  Machine conflicts: {machine_conflicts}")
    
    if machine_conflicts == 0:
        print("  ✅ PERFECT: No machine conflicts")
        validation_results.append(("Machines", "PASS", 10))
    else:
        print("  ❌ FAIL: Machine conflicts detected")
        validation_results.append(("Machines", "FAIL", 0))
    
    # ========================
    # CHECK 4: MAKESPAN EFFICIENCY
    # ========================
    print("\n✓ CHECK 4: Makespan and Efficiency")
    print("-" * 40)
    
    # Calculate makespan
    makespan = max(end for _, end, _ in env.task_schedules.values()) if env.task_schedules else 0
    
    # Theoretical minimum makespan
    total_processing = sum(t.processing_time for t in env.loader.tasks)
    n_machines = len(env.loader.machines)
    theoretical_min = total_processing / n_machines
    
    efficiency = (theoretical_min / makespan * 100) if makespan > 0 else 0
    
    print(f"  Actual makespan: {makespan:.1f} hours ({makespan/24:.1f} days)")
    print(f"  Theoretical min: {theoretical_min:.1f} hours")
    print(f"  Efficiency: {efficiency:.1f}%")
    
    if efficiency >= 50:
        print("  ✅ EXCELLENT: High efficiency")
        validation_results.append(("Efficiency", "PASS", 10))
    elif efficiency >= 30:
        print("  ✅ GOOD: Reasonable efficiency")
        validation_results.append(("Efficiency", "PASS", 7))
    elif efficiency >= 10:
        print("  ⚠️  LOW: Poor efficiency")
        validation_results.append(("Efficiency", "WARN", 4))
    else:
        print("  ❌ VERY LOW: Terrible efficiency")
        validation_results.append(("Efficiency", "FAIL", 0))
    
    # ========================
    # CHECK 5: ON-TIME DELIVERY
    # ========================
    print("\n✓ CHECK 5: On-Time Delivery Performance")
    print("-" * 40)
    
    late_jobs = 0
    very_late_jobs = 0
    early_jobs = 0
    
    for task_idx, (start, end, machine) in env.task_schedules.items():
        task = env.loader.tasks[task_idx]
        family = env.loader.families[task.family_id]
        lcd_hours = family.lcd_days_remaining * 24
        
        time_diff = lcd_hours - end
        
        if time_diff < 0:
            late_jobs += 1
            if time_diff < -24:
                very_late_jobs += 1
        elif time_diff > 72:
            early_jobs += 1
    
    on_time_rate = 1 - (late_jobs / info['tasks_scheduled']) if info['tasks_scheduled'] > 0 else 0
    
    print(f"  Early (>3 days): {early_jobs} ({early_jobs/info['tasks_scheduled']*100:.1f}%)")
    print(f"  On-time: {info['tasks_scheduled']-late_jobs} ({on_time_rate*100:.1f}%)")
    print(f"  Late: {late_jobs} ({late_jobs/info['tasks_scheduled']*100:.1f}%)")
    print(f"  Very late (>24h): {very_late_jobs}")
    
    if on_time_rate >= 0.80:
        print("  ✅ EXCELLENT: High on-time rate")
        validation_results.append(("Delivery", "PASS", 10))
    elif on_time_rate >= 0.60:
        print("  ✅ GOOD: Acceptable on-time rate")
        validation_results.append(("Delivery", "PASS", 7))
    elif on_time_rate >= 0.40:
        print("  ⚠️  POOR: Low on-time rate")
        validation_results.append(("Delivery", "WARN", 4))
    else:
        print("  ❌ FAIL: Very low on-time rate")
        validation_results.append(("Delivery", "FAIL", 2))
    
    # ========================
    # CHECK 6: MACHINE UTILIZATION
    # ========================
    print("\n✓ CHECK 6: Machine Utilization Balance")
    print("-" * 40)
    
    machine_usage = {}
    for task_idx, (start, end, machine) in env.task_schedules.items():
        if machine not in machine_usage:
            machine_usage[machine] = 0
        machine_usage[machine] += (end - start)
    
    if machine_usage:
        avg_usage = np.mean(list(machine_usage.values()))
        std_usage = np.std(list(machine_usage.values()))
        cv = (std_usage / avg_usage * 100) if avg_usage > 0 else 0
        
        print(f"  Machines used: {len(machine_usage)}/{n_machines}")
        print(f"  Average usage: {avg_usage:.1f} hours")
        print(f"  Std deviation: {std_usage:.1f} hours")
        print(f"  Coefficient of variation: {cv:.1f}%")
        
        if cv < 50 and len(machine_usage) > n_machines * 0.5:
            print("  ✅ GOOD: Balanced machine usage")
            validation_results.append(("Balance", "PASS", 8))
        elif cv < 100:
            print("  ⚠️  MODERATE: Some imbalance")
            validation_results.append(("Balance", "WARN", 5))
        else:
            print("  ❌ POOR: Very imbalanced")
            validation_results.append(("Balance", "FAIL", 2))
    
    # ========================
    # CHECK 7: PERFORMANCE SPEED
    # ========================
    print("\n✓ CHECK 7: Scheduling Speed")
    print("-" * 40)
    
    tasks_per_second = info['tasks_scheduled'] / scheduling_time if scheduling_time > 0 else 0
    
    print(f"  Total time: {scheduling_time:.2f} seconds")
    print(f"  Tasks/second: {tasks_per_second:.1f}")
    print(f"  Steps taken: {steps}")
    
    if tasks_per_second >= 10:
        print("  ✅ FAST: Excellent speed")
        validation_results.append(("Speed", "PASS", 10))
    elif tasks_per_second >= 5:
        print("  ✅ GOOD: Reasonable speed")
        validation_results.append(("Speed", "PASS", 7))
    else:
        print("  ⚠️  SLOW: Could be faster")
        validation_results.append(("Speed", "WARN", 4))
    
    # ========================
    # FINAL SCORING
    # ========================
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total_score = 0
    max_score = 0
    failed_checks = []
    
    print(f"\n{'Check':<15} {'Result':<10} {'Score':<10}")
    print("-" * 40)
    
    for check_name, result, score in validation_results:
        total_score += score
        max_score += 10
        
        if result == "FAIL":
            failed_checks.append(check_name)
            emoji = "❌"
        elif result == "WARN":
            emoji = "⚠️"
        else:
            emoji = "✅"
        
        print(f"{check_name:<15} {emoji} {result:<8} {score:>2}/10")
    
    final_percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    print("-" * 40)
    print(f"{'TOTAL':<15} {'':10} {total_score:>2}/{max_score}")
    print(f"{'PERCENTAGE':<15} {'':10} {final_percentage:>5.1f}%")
    
    # ========================
    # FINAL VERDICT
    # ========================
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if failed_checks:
        print(f"\n❌ CRITICAL FAILURES in: {', '.join(failed_checks)}")
    
    if final_percentage >= 90:
        print("\n🏆 EXCELLENT MODEL!")
        print("This is a HIGH-QUALITY scheduling model.")
        print("Ready for production use.")
    elif final_percentage >= 70:
        print("\n✅ GOOD MODEL")
        print("Model performs well with minor issues.")
        print("Suitable for most scheduling tasks.")
    elif final_percentage >= 50:
        print("\n⚠️  ACCEPTABLE MODEL")
        print("Model works but has room for improvement.")
        print("Consider additional training.")
    else:
        print("\n❌ POOR MODEL")
        print("Model needs significant improvement.")
        print("Recommend retraining with better hyperparameters.")
    
    print("\n" + "="*80)
    print(f"Model Score: {final_percentage:.1f}%")
    print("="*80)
    
    return final_percentage >= 70  # Return True if model is good

if __name__ == "__main__":
    is_good = validate_model_performance()
    exit(0 if is_good else 1)