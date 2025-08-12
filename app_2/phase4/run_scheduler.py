#!/usr/bin/env python3
"""
Simple scheduler using Phase 4 models - production ready version.
"""

import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from phase4.environments.small_balanced_env import SmallBalancedEnvironment


def schedule_jobs(model_path=None):
    """Schedule jobs using trained model or fallback."""
    
    # Load environment with real data
    data_file = Path(__file__).parent / "data" / "small_balanced_data.json"
    env = SmallBalancedEnvironment(str(data_file))
    
    print(f"Environment loaded: {env.total_tasks} tasks, {len(env.machines)} machines")
    
    # Try to load model if path provided
    if model_path and Path(model_path).exists():
        print(f"Loading model: {model_path}")
        model = PPO.load(model_path)
        use_model = True
    else:
        print("No model provided, using simple scheduler")
        model = None
        use_model = False
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = 500
    scheduled_count = 0
    
    print("\nStarting scheduling...")
    
    while not done and steps < max_steps:
        if use_model:
            # Use trained model
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Simple scheduler: try random valid actions
            found_valid = False
            for _ in range(100):  # Try 100 random actions
                action = env.action_space.sample()
                job_idx = action[0]
                
                # Check if job not already scheduled
                if job_idx not in env.scheduled_jobs:
                    found_valid = True
                    break
            
            if not found_valid:
                break
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        if info.get('action_valid', False):
            scheduled_count += 1
            print(f"Step {steps}: Scheduled task (total: {scheduled_count}/{env.total_tasks})")
    
    # Report results
    print(f"\n{'='*50}")
    print("SCHEDULING COMPLETE")
    print(f"{'='*50}")
    print(f"Tasks scheduled: {len(env.scheduled_jobs)}/{env.total_tasks}")
    print(f"Completion rate: {len(env.scheduled_jobs)/env.total_tasks*100:.1f}%")
    print(f"Steps taken: {steps}")
    
    # Extract schedule details
    schedule = []
    for job_idx, assignment in env.job_assignments.items():
        schedule.append({
            'job_id': job_idx,
            'machine_id': assignment['machine_id'],
            'start_time': assignment['start_time'],
            'duration': assignment['duration']
        })
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"schedule_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_tasks': env.total_tasks,
            'scheduled_tasks': len(env.scheduled_jobs),
            'completion_rate': len(env.scheduled_jobs)/env.total_tasks*100,
            'schedule': schedule
        }, f, indent=2)
    
    print(f"\nSchedule saved to: {output_file}")
    
    return schedule, env


if __name__ == "__main__":
    # Check if model path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Use latest balanced model
        model_dir = Path(__file__).parent / "results" / "balanced" / "checkpoints"
        if model_dir.exists():
            models = sorted(model_dir.glob("*.zip"))
            if models:
                model_path = str(models[-1])
                print(f"Using latest model: {model_path}")
            else:
                model_path = None
        else:
            model_path = None
    
    schedule, env = schedule_jobs(model_path)