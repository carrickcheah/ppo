"""Visualization utilities for PPO scheduling results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path


def plot_training_curves(
    log_path: str,
    save_path: Optional[str] = None,
    window_size: int = 100
) -> None:
    """Plot training curves from tensorboard logs.
    
    Args:
        log_path: Path to training logs
        save_path: Path to save the plot
        window_size: Window size for moving average
    """
    try:
        from stable_baselines3.common.results_plotter import load_results, ts2xy
        
        # Load results
        results_dir = Path(log_path).parent / 'eval'
        if results_dir.exists():
            df = load_results(str(results_dir))
            x, y = ts2xy(df, 'timesteps')
            
            # Plot rewards
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Episode rewards
            ax1.plot(x, y, alpha=0.3, label='Raw rewards')
            
            # Moving average
            if len(y) > window_size:
                moving_avg = pd.Series(y).rolling(window=window_size).mean()
                ax1.plot(x, moving_avg, label=f'Moving avg ({window_size} eps)')
            
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Rewards Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Episode lengths
            if 'l' in df.columns:
                ep_lengths = df['l'].values
                x_eps = np.arange(len(ep_lengths))
                ax2.plot(x_eps, ep_lengths, alpha=0.5)
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Episode Length')
                ax2.set_title('Episode Lengths Over Training')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
    except ImportError:
        print("Could not load training curves. Make sure stable-baselines3 is installed.")
    except Exception as e:
        print(f"Error loading training curves: {e}")


def visualize_policy_heatmap(
    model,
    env,
    n_samples: int = 1000,
    save_path: Optional[str] = None
) -> None:
    """Visualize policy as heatmap showing action probabilities in different states.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        n_samples: Number of state samples to collect
        save_path: Path to save the plot
    """
    # Collect states and action probabilities
    states = []
    action_probs = []
    
    for _ in range(n_samples // 10):
        obs, _ = env.reset()
        for _ in range(10):
            # Get action probabilities
            action, _ = model.predict(obs, deterministic=False)
            
            # Get the policy network
            if hasattr(model.policy, 'action_net'):
                # Get action logits
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                with model.policy.sess.as_default():
                    features = model.policy._get_features(obs_tensor)
                    logits = model.policy.action_net(features)
                    probs = model.policy._softmax(logits).numpy()[0]
            else:
                # For newer versions of SB3
                import torch
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs[0].numpy()
            
            states.append(obs)
            action_probs.append(probs)
            
            # Take action
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    
    # Convert to arrays
    states = np.array(states)
    action_probs = np.array(action_probs)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Average action probabilities by state features
    # Group by number of scheduled jobs
    n_jobs = env.n_jobs
    job_counts = np.sum(states[:, env.n_machines:env.n_machines+n_jobs], axis=1).astype(int)
    
    # Create heatmap data
    heatmap_data = np.zeros((n_jobs + 1, env.action_space.n))
    counts = np.zeros(n_jobs + 1)
    
    for i, count in enumerate(job_counts):
        heatmap_data[count] += action_probs[i]
        counts[count] += 1
    
    # Normalize
    for i in range(n_jobs + 1):
        if counts[i] > 0:
            heatmap_data[i] /= counts[i]
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=[f'Job {i}' for i in range(n_jobs)] + ['Wait'],
        yticklabels=[f'{i} jobs scheduled' for i in range(n_jobs + 1)],
        ax=ax
    )
    
    ax.set_title('Policy Heatmap: Action Probabilities by State')
    ax.set_xlabel('Action')
    ax.set_ylabel('Number of Jobs Already Scheduled')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_schedule_gantt(
    env,
    model=None,
    use_random: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """Visualize job schedule as Gantt chart.
    
    Args:
        env: Environment instance
        model: Trained model (if None, uses random policy)
        use_random: If True, use random policy instead of model
        save_path: Path to save the plot
        
    Returns:
        Dictionary with metrics (makespan, utilization)
    """
    # Reset environment
    obs, _ = env.reset()
    
    # Track job assignments
    job_assignments = []  # (job_id, machine_id, start_time, duration)
    machine_times = np.zeros(env.n_machines)
    
    # Run episode
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < env.max_episode_steps:
        if use_random or model is None:
            # Random valid action
            action_mask = env.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        else:
            # Model action
            action, _ = model.predict(obs, deterministic=True)
        
        # Track state before action
        if action < env.n_jobs and not env.job_scheduled[action]:
            # Find which machine will be assigned (minimum load)
            machine_id = np.argmin(env.machine_loads)
            start_time = machine_times[machine_id]
            duration = env.job_times[action]
            
            job_assignments.append((action, machine_id, start_time, duration))
            machine_times[machine_id] += duration
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color map for jobs
    colors = plt.cm.Set3(np.linspace(0, 1, env.n_jobs))
    
    # Plot jobs
    for job_id, machine_id, start_time, duration in job_assignments:
        rect = Rectangle(
            (start_time, machine_id - 0.4),
            duration,
            0.8,
            facecolor=colors[job_id],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
        # Add job label
        ax.text(
            start_time + duration/2,
            machine_id,
            f'J{job_id}',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Set axis properties
    ax.set_ylim(-0.5, env.n_machines - 0.5)
    ax.set_xlim(0, max(machine_times) * 1.1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(env.n_machines))
    ax.set_yticklabels([f'Machine {i}' for i in range(env.n_machines)])
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Calculate metrics
    makespan = max(machine_times) if len(job_assignments) > 0 else 0
    utilization = np.mean(machine_times) / makespan if makespan > 0 else 0
    
    # Add title with metrics
    policy_type = "Random" if use_random or model is None else "Trained"
    ax.set_title(
        f'{policy_type} Policy Schedule - Makespan: {makespan:.1f}, '
        f'Utilization: {utilization:.2%}, Reward: {total_reward:.1f}'
    )
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors[i], label=f'Job {i} (time={env.job_times[i]})')
        for i in range(env.n_jobs)
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'makespan': makespan,
        'utilization': utilization,
        'total_reward': total_reward,
        'jobs_scheduled': len(job_assignments)
    }


def compare_policies(
    env,
    model,
    n_episodes: int = 20,
    save_path: Optional[str] = None
) -> None:
    """Compare trained policy vs random policy performance.
    
    Args:
        env: Environment instance
        model: Trained model
        n_episodes: Number of episodes to evaluate
        save_path: Path to save the plot
    """
    # Evaluate both policies
    trained_metrics = {
        'rewards': [],
        'makespans': [],
        'utilizations': [],
        'steps': []
    }
    
    random_metrics = {
        'rewards': [],
        'makespans': [],
        'utilizations': [],
        'steps': []
    }
    
    for _ in range(n_episodes):
        # Trained policy
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < env.max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        trained_metrics['rewards'].append(total_reward)
        trained_metrics['steps'].append(steps)
        if 'final_makespan' in info:
            trained_metrics['makespans'].append(info['final_makespan'])
            trained_metrics['utilizations'].append(info['final_utilization'])
        
        # Random policy
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < env.max_episode_steps:
            action_mask = env.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        random_metrics['rewards'].append(total_reward)
        random_metrics['steps'].append(steps)
        if 'final_makespan' in info:
            random_metrics['makespans'].append(info['final_makespan'])
            random_metrics['utilizations'].append(info['final_utilization'])
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rewards comparison
    ax1.boxplot([random_metrics['rewards'], trained_metrics['rewards']], 
                labels=['Random', 'Trained'])
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Makespan comparison
    if trained_metrics['makespans']:
        ax2.boxplot([random_metrics['makespans'], trained_metrics['makespans']], 
                    labels=['Random', 'Trained'])
        ax2.set_ylabel('Makespan')
        ax2.set_title('Makespan Comparison')
        ax2.grid(True, alpha=0.3)
    
    # Utilization comparison
    if trained_metrics['utilizations']:
        ax3.boxplot([random_metrics['utilizations'], trained_metrics['utilizations']], 
                    labels=['Random', 'Trained'])
        ax3.set_ylabel('Utilization')
        ax3.set_title('Machine Utilization Comparison')
        ax3.grid(True, alpha=0.3)
    
    # Steps comparison
    ax4.boxplot([random_metrics['steps'], trained_metrics['steps']], 
                labels=['Random', 'Trained'])
    ax4.set_ylabel('Steps to Complete')
    ax4.set_title('Episode Length Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Add summary statistics
    fig.suptitle(
        f'Policy Comparison over {n_episodes} episodes\n'
        f'Trained: Avg Reward = {np.mean(trained_metrics["rewards"]):.1f}, '
        f'Random: Avg Reward = {np.mean(random_metrics["rewards"]):.1f}'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n=== Performance Comparison ===")
    print(f"Metric{'':20} Random{'':10} Trained{'':10} Improvement")
    print("-" * 60)
    
    for metric in ['rewards', 'makespans', 'utilizations']:
        if metric in trained_metrics and trained_metrics[metric]:
            random_mean = np.mean(random_metrics[metric])
            trained_mean = np.mean(trained_metrics[metric])
            
            if metric == 'makespans':
                # Lower is better for makespan
                improvement = (random_mean - trained_mean) / random_mean * 100
            else:
                # Higher is better for rewards and utilization
                improvement = (trained_mean - random_mean) / random_mean * 100
            
            print(f"{metric.capitalize():<25} {random_mean:>10.2f} {trained_mean:>10.2f} "
                  f"{improvement:>10.1f}%")