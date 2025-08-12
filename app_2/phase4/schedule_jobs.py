#!/usr/bin/env python3
"""
Schedule jobs using Phase 4 trained PPO models.
Despite low training performance, demonstrates model usage for scheduling.
"""

import json
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from environments.small_balanced_env import SmallBalancedEnvironment
from environments.small_rush_env import SmallRushEnvironment
from environments.small_bottleneck_env import SmallBottleneckEnvironment
from environments.small_complex_env import SmallComplexEnvironment


def load_latest_model(strategy="balanced"):
    """Load the latest trained model for a strategy."""
    results_dir = Path(__file__).parent / "results" / strategy / "checkpoints"
    
    if not results_dir.exists():
        print(f"No models found for strategy: {strategy}")
        return None
    
    # Find the latest model
    model_files = sorted(results_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not model_files:
        print(f"No model files found in {results_dir}")
        return None
    
    latest_model = model_files[0]
    print(f"Loading model: {latest_model.name}")
    
    try:
        model = PPO.load(str(latest_model))
        return model, latest_model.stem
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_environment(strategy="balanced"):
    """Create the appropriate environment for the strategy."""
    data_dir = Path(__file__).parent / "data"
    
    if strategy == "balanced":
        data_file = data_dir / "small_balanced_data.json"
        env = SmallBalancedEnvironment(str(data_file))
    elif strategy == "rush":
        data_file = data_dir / "small_rush_data.json"
        env = SmallRushEnvironment(str(data_file))
    elif strategy == "bottleneck":
        data_file = data_dir / "small_bottleneck_data.json"
        env = SmallBottleneckEnvironment(str(data_file))
    elif strategy == "complex":
        data_file = data_dir / "small_complex_data.json"
        env = SmallComplexEnvironment(str(data_file))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return env


def schedule_with_model(model, env):
    """Use the PPO model to generate a schedule."""
    obs, _ = env.reset()
    done = False
    schedule = []
    steps = 0
    max_steps = 1000  # Prevent infinite loops
    
    print("\nGenerating schedule with PPO model...")
    
    while not done and steps < max_steps:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Track scheduled jobs
        if info.get('action_valid', False):
            last_scheduled = info.get('last_scheduled')
            if last_scheduled:
                schedule.append(last_scheduled)
                print(f"Step {steps}: Scheduled {last_scheduled['job_id']} on machine {last_scheduled['machine_id']}")
    
    print(f"\nScheduling completed in {steps} steps")
    print(f"Jobs scheduled: {len(schedule)}/{len(env.jobs)}")
    
    return schedule, env


def fallback_scheduler(env):
    """Simple FIFO scheduler as fallback when model fails."""
    print("\nUsing fallback FIFO scheduler...")
    
    obs, _ = env.reset()
    schedule = []
    done = False
    steps = 0
    max_steps = 1000
    
    while not done and steps < max_steps:
        # Try to schedule jobs in order
        valid_action_found = False
        
        for job_idx in range(len(env.jobs)):
            if env.scheduled_jobs[job_idx]:
                continue
                
            # Try each machine
            for machine_idx in range(len(env.machines)):
                action = np.array([job_idx, machine_idx])
                
                # Check if action is valid
                if env._is_action_valid(job_idx, machine_idx):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                    
                    if info.get('action_valid', False):
                        last_scheduled = info.get('last_scheduled')
                        if last_scheduled:
                            schedule.append(last_scheduled)
                            print(f"Step {steps}: Scheduled {last_scheduled['job_id']} on machine {last_scheduled['machine_id']}")
                    
                    valid_action_found = True
                    break
            
            if valid_action_found:
                break
        
        if not valid_action_found:
            print("No more valid actions available")
            break
    
    print(f"\nFallback scheduling completed in {steps} steps")
    print(f"Jobs scheduled: {len(schedule)}/{len(env.jobs)}")
    
    return schedule, env


def visualize_schedule(schedule, env, title="Schedule Gantt Chart"):
    """Create a Gantt chart visualization of the schedule."""
    if not schedule:
        print("No jobs scheduled to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Color map for different jobs
    colors = plt.cm.Set3(np.linspace(0, 1, len(set(s['job_id'] for s in schedule))))
    job_colors = {job_id: colors[i] for i, job_id in enumerate(set(s['job_id'] for s in schedule))}
    
    # Plot scheduled tasks
    machine_names = [m['name'] for m in env.machines]
    y_pos = {name: i for i, name in enumerate(machine_names)}
    
    for task in schedule:
        machine_name = next(m['name'] for m in env.machines if m['id'] == task['machine_id'])
        y = y_pos[machine_name]
        
        # Create rectangle for task
        rect = patches.Rectangle(
            (task['start_time'], y - 0.4),
            task['processing_time'],
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=job_colors[task['job_id']],
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add job label
        if task['processing_time'] > 2:  # Only add label if task is wide enough
            ax.text(
                task['start_time'] + task['processing_time']/2,
                y,
                task['job_id'][:10],
                ha='center',
                va='center',
                fontsize=8
            )
    
    # Configure plot
    ax.set_ylim(-0.5, len(machine_names) - 0.5)
    ax.set_yticks(range(len(machine_names)))
    ax.set_yticklabels(machine_names)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Machines')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add current time line
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Start')
    
    # Calculate makespan
    if schedule:
        makespan = max(s['start_time'] + s['processing_time'] for s in schedule)
        ax.set_xlim(0, makespan * 1.1)
        ax.axvline(x=makespan, color='green', linestyle='--', alpha=0.5, label=f'Makespan: {makespan:.1f}h')
    
    ax.legend(loc='upper right')
    
    # Add statistics
    stats_text = f"Jobs Scheduled: {len(schedule)}/{len(env.jobs)}\n"
    stats_text += f"Machines Used: {len(set(s['machine_id'] for s in schedule))}/{len(env.machines)}\n"
    
    if schedule:
        utilization = sum(s['processing_time'] for s in schedule) / (makespan * len(env.machines)) * 100
        stats_text += f"Utilization: {utilization:.1f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """Main scheduling function."""
    print("PPO Job Scheduling System")
    print("=" * 50)
    
    # Select strategy
    strategies = ["balanced", "rush", "bottleneck", "complex"]
    print("\nAvailable strategies:")
    for i, s in enumerate(strategies, 1):
        print(f"{i}. {s}")
    
    choice = input("\nSelect strategy (1-4) or press Enter for 'balanced': ").strip()
    
    if choice == "":
        strategy = "balanced"
    elif choice.isdigit() and 1 <= int(choice) <= 4:
        strategy = strategies[int(choice) - 1]
    else:
        print("Invalid choice, using 'balanced'")
        strategy = "balanced"
    
    print(f"\nUsing strategy: {strategy}")
    
    # Load model
    model_result = load_latest_model(strategy)
    
    # Create environment
    env = create_environment(strategy)
    
    # Generate schedule
    if model_result:
        model, model_name = model_result
        print(f"\nUsing PPO model: {model_name}")
        schedule, env = schedule_with_model(model, env)
        
        # If model fails to schedule enough jobs, use fallback
        if len(schedule) < len(env.jobs) * 0.5:  # Less than 50% scheduled
            print("\nModel performance insufficient, switching to fallback scheduler...")
            env.reset()
            schedule, env = fallback_scheduler(env)
    else:
        print("\nNo trained model available, using fallback scheduler...")
        schedule, env = fallback_scheduler(env)
    
    # Display results
    print("\n" + "=" * 50)
    print("SCHEDULING RESULTS")
    print("=" * 50)
    
    if schedule:
        print(f"\nSuccessfully scheduled {len(schedule)} jobs:")
        for i, task in enumerate(schedule[:10], 1):  # Show first 10
            print(f"{i}. {task['job_id']} on {task['machine_id']} "
                  f"at time {task['start_time']:.1f}h for {task['processing_time']:.1f}h")
        
        if len(schedule) > 10:
            print(f"... and {len(schedule) - 10} more jobs")
        
        # Calculate metrics
        makespan = max(s['start_time'] + s['processing_time'] for s in schedule)
        total_processing = sum(s['processing_time'] for s in schedule)
        utilization = total_processing / (makespan * len(env.machines)) * 100
        
        print(f"\nPerformance Metrics:")
        print(f"- Makespan: {makespan:.1f} hours")
        print(f"- Machine Utilization: {utilization:.1f}%")
        print(f"- Completion Rate: {len(schedule)/len(env.jobs)*100:.1f}%")
        
        # Save schedule to file
        output_dir = Path(__file__).parent / "schedules"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"schedule_{strategy}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'strategy': strategy,
                'timestamp': timestamp,
                'schedule': schedule,
                'metrics': {
                    'makespan': makespan,
                    'utilization': utilization,
                    'completion_rate': len(schedule)/len(env.jobs)*100,
                    'jobs_scheduled': len(schedule),
                    'total_jobs': len(env.jobs)
                }
            }, f, indent=2)
        
        print(f"\nSchedule saved to: {output_file}")
        
        # Visualize schedule
        fig = visualize_schedule(schedule, env, f"{strategy.capitalize()} Strategy Schedule")
        
        viz_file = output_dir / f"gantt_{strategy}_{timestamp}.png"
        fig.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"Gantt chart saved to: {viz_file}")
        
        plt.show()
    else:
        print("\nNo jobs were scheduled successfully")
    
    print("\n" + "=" * 50)
    print("Scheduling complete!")


if __name__ == "__main__":
    main()