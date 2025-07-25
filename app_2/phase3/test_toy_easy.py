"""
Test Toy Easy stage to verify 100% utilization is achievable
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.curriculum_env import CurriculumSchedulingEnv


def test_manual_scheduling():
    """Test if 100% utilization is achievable with manual scheduling."""
    print("=== TESTING TOY EASY STAGE ===")
    print("Goal: Achieve 100% utilization with 5 jobs, 3 machines\n")
    
    # Create Toy Easy environment
    stage_config = {
        'name': 'toy_easy',
        'jobs': 5,
        'machines': 3,
        'description': 'Learn sequence rules',
        'multi_machine_ratio': 0.0
    }
    
    env = CurriculumSchedulingEnv(
        stage_config=stage_config,
        data_source="synthetic",
        reward_profile="learning",
        seed=42
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment initialized with {env.n_jobs} jobs and {env.n_machines} machines")
    
    # Show job details
    print("\nJob Details:")
    for job_id, state in env.job_states.items():
        total_time = sum(task['processing_time'] for task in state['tasks'])
        print(f"  {job_id}: {state['total_sequences']} sequences, "
              f"total time: {total_time:.1f}h, LCD: {state['lcd_hours']:.0f}h")
        for task in state['tasks']:
            print(f"    Seq {task['sequence']}: {task['processing_time']:.1f}h on machines {task['capable_machines']}")
    
    # Calculate theoretical minimum makespan
    total_work = sum(
        sum(task['processing_time'] for task in state['tasks'])
        for state in env.job_states.values()
    )
    theoretical_min = total_work / env.n_machines
    print(f"\nTotal work: {total_work:.1f} hours")
    print(f"Theoretical minimum makespan: {theoretical_min:.1f} hours")
    
    # Manual scheduling strategy: Schedule jobs in order
    print("\n--- Starting Manual Scheduling ---")
    done = False
    step = 0
    actions_taken = []
    
    while not done:
        # Find next job to schedule
        scheduled = False
        
        for job_idx, job_id in enumerate(env.job_ids):
            job_state = env.job_states[job_id]
            
            # Skip completed jobs
            if job_state['status'] == 'completed':
                continue
            
            # Skip in-progress jobs
            if job_state['status'] == 'in_progress':
                continue
            
            # Get current task
            current_seq = job_state['current_sequence']
            if current_seq > job_state['total_sequences']:
                continue
            
            current_task = job_state['tasks'][current_seq - 1]
            
            # Try each capable machine
            for machine_id in current_task['capable_machines']:
                machine_idx = env.machine_id_to_idx[machine_id]
                
                # Test action
                action = np.array([job_idx, machine_idx])
                
                # Check if valid
                is_valid, reason = env._is_valid_action(job_id, machine_id)
                
                if is_valid:
                    # Execute action
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    print(f"\nStep {step}: Scheduled {job_id} seq {current_seq} on machine {machine_id}")
                    print(f"  Time: {env.current_time:.1f}h, Reward: {reward:.2f}")
                    print(f"  Valid: {info.get('action_valid', False)}")
                    
                    actions_taken.append({
                        'step': step,
                        'job': job_id,
                        'sequence': current_seq,
                        'machine': machine_id,
                        'time': env.current_time,
                        'reward': reward
                    })
                    
                    scheduled = True
                    break
            
            if scheduled:
                break
        
        if not scheduled and not done:
            # No valid actions, find next event time
            next_event_times = []
            for machine_id, machine_state in env.machine_states.items():
                if machine_state['available_at'] > env.current_time:
                    next_event_times.append(machine_state['available_at'])
            
            if next_event_times:
                # Jump to next machine availability
                env.current_time = min(next_event_times)
            else:
                # All machines free, advance by small amount
                env.current_time += 0.1
            
            if env.current_time >= env.max_time:
                done = True
        
        step += 1
        
        # Safety check
        if step > 100:
            print("\nToo many steps, stopping...")
            break
    
    # Final metrics
    env._calculate_final_metrics()
    metrics = env.episode_metrics
    
    print("\n=== RESULTS ===")
    print(f"Jobs completed: {metrics['jobs_completed']}/{env.n_jobs}")
    print(f"Sequences completed: {metrics['sequences_completed']}/{metrics['total_sequences']}")
    print(f"Machine utilization: {metrics['machine_utilization']:.1%}")
    print(f"Makespan: {metrics.get('makespan', env.current_time):.1f} hours")
    print(f"Jobs late: {metrics['jobs_late']}")
    print(f"Total reward: {sum(a['reward'] for a in actions_taken):.2f}")
    
    # Visualize schedule
    visualize_schedule(env, actions_taken)
    
    return metrics['machine_utilization']


def visualize_schedule(env, actions_taken):
    """Create a Gantt chart of the schedule."""
    plt.figure(figsize=(12, 6))
    
    # Colors for different jobs
    colors = plt.cm.tab10(np.linspace(0, 1, env.n_jobs))
    job_colors = {job_id: colors[i] for i, job_id in enumerate(env.job_ids)}
    
    # Plot schedule
    for job_id, job_state in env.job_states.items():
        if job_state['completion_times']:
            # Get all task completions for this job
            for seq_idx, end_time in enumerate(job_state['completion_times']):
                task = job_state['tasks'][seq_idx]
                processing_time = task['processing_time']
                start_time = end_time - processing_time
                
                # Plot on each machine used
                for machine_id in task['capable_machines']:
                    machine_idx = env.machine_id_to_idx[machine_id]
                    
                    # Check if this machine was actually used (simplified)
                    plt.barh(
                        machine_idx,
                        processing_time,
                        left=start_time,
                        height=0.8,
                        color=job_colors[job_id],
                        edgecolor='black',
                        label=f"{job_id} seq{seq_idx+1}" if seq_idx == 0 else ""
                    )
                    
                    # Add text
                    plt.text(
                        start_time + processing_time/2,
                        machine_idx,
                        f"{job_id}\nS{seq_idx+1}",
                        ha='center',
                        va='center',
                        fontsize=8
                    )
    
    # Format plot
    plt.yticks(range(env.n_machines), [f"Machine {m}" for m in env.machine_ids])
    plt.xlabel('Time (hours)')
    plt.title('Toy Easy Schedule Visualization')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(env.current_time, 20))
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('/Users/carrickcheah/Project/ppo/app_2/visualizations/phase_3', exist_ok=True)
    plt.savefig('/Users/carrickcheah/Project/ppo/app_2/visualizations/phase_3/toy_easy_manual_schedule.png', dpi=150)
    print("\nSchedule visualization saved to: /Users/carrickcheah/Project/ppo/app_2/visualizations/phase_3/toy_easy_manual_schedule.png")
    plt.close()


def test_random_actions():
    """Test with random valid actions to see average performance."""
    print("\n\n=== TESTING RANDOM ACTIONS ===")
    
    stage_config = {
        'name': 'toy_easy',
        'jobs': 5,
        'machines': 3,
        'description': 'Learn sequence rules',
        'multi_machine_ratio': 0.0
    }
    
    utilizations = []
    rewards = []
    
    for seed in range(5):
        env = CurriculumSchedulingEnv(
            stage_config=stage_config,
            data_source="synthetic",
            reward_profile="learning",
            seed=seed
        )
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        env._calculate_final_metrics()
        utilizations.append(env.episode_metrics['machine_utilization'])
        rewards.append(total_reward)
    
    print(f"Average utilization: {np.mean(utilizations):.1%} ± {np.std(utilizations):.1%}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    # Test manual scheduling
    utilization = test_manual_scheduling()
    
    if utilization >= 0.95:
        print("\n✓ SUCCESS: Achieved 95%+ utilization manually!")
        print("This proves the environment can reach high utilization.")
    else:
        print("\n✗ WARNING: Could not achieve 95%+ utilization manually.")
        print("Environment or reward structure may need adjustment.")
    
    # Test random actions
    test_random_actions()