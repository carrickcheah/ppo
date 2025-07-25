"""
Test Small Rush Stage Performance

Comprehensive testing of the small_rush model.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from phase3.environments.curriculum_env import CurriculumSchedulingEnv


def test_small_rush_model():
    """Test the small_rush model comprehensively."""
    print("=" * 80)
    print("SMALL RUSH MODEL TESTING")
    print("=" * 80)
    print()
    
    # Stage configuration
    stage_config = {
        'name': 'Small Rush - Handle Urgency',
        'jobs': 50,
        'machines': 30,
        'data_source': 'snapshot_rush',
        'learning_rate': 0.001,
        'n_steps': 256,
        'batch_size': 32,
        'timesteps': 50000
    }
    
    # Load model
    model_path = 'phase3/checkpoints/small_rush_final.zip'
    vec_norm_path = 'phase3/checkpoints/small_rush_vec_normalize.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from: {model_path}")
    
    # Create environment
    snapshot_path = 'phase3/snapshots/snapshot_rush.json'
    env = CurriculumSchedulingEnv(
        stage_config=stage_config,
        snapshot_path=snapshot_path,
        reward_profile='balanced',
        seed=42
    )
    
    # Wrap in vec env
    vec_env = DummyVecEnv([lambda: env])
    
    # Load vec normalize if exists
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        
    # Load model
    model = PPO.load(model_path, env=vec_env)
    
    # Test metrics
    n_episodes = 20
    results = {
        'episodes': [],
        'rewards': [],
        'late_jobs': [],
        'completed_jobs': [],
        'utilization': [],
        'makespan': [],
        'schedule_quality': []
    }
    
    print(f"\nRunning {n_episodes} test episodes...")
    print("-" * 60)
    
    for episode in range(n_episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            steps += 1
            
        # Extract metrics
        if 'episode_metrics' in info[0]:
            metrics = info[0]['episode_metrics']
            
            results['episodes'].append(episode)
            results['rewards'].append(episode_reward)
            results['late_jobs'].append(metrics.get('jobs_late', 0))
            results['completed_jobs'].append(metrics.get('jobs_completed', 0))
            results['utilization'].append(metrics.get('machine_utilization', 0))
            results['makespan'].append(metrics.get('makespan', 0))
            
            # Calculate schedule quality
            quality = 100 * (1 - metrics.get('jobs_late', 0) / max(1, metrics.get('jobs_completed', 1)))
            results['schedule_quality'].append(quality)
            
            print(f"Episode {episode+1:2d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Steps: {steps:3d} | "
                  f"Completed: {metrics.get('jobs_completed', 0):2d} | "
                  f"Late: {metrics.get('jobs_late', 0):2d} | "
                  f"Util: {metrics.get('machine_utilization', 0)*100:5.1f}%")
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    avg_reward = np.mean(results['rewards'])
    std_reward = np.std(results['rewards'])
    avg_late = np.mean(results['late_jobs'])
    avg_completed = np.mean(results['completed_jobs'])
    avg_util = np.mean(results['utilization']) * 100
    avg_quality = np.mean(results['schedule_quality'])
    
    print(f"\nAverage Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Completed Jobs: {avg_completed:.1f} / 30")
    print(f"Average Late Jobs: {avg_late:.1f}")
    print(f"Average Utilization: {avg_util:.1f}%")
    print(f"Schedule Quality: {avg_quality:.1f}%")
    
    # Performance assessment
    print("\n" + "-" * 60)
    print("PERFORMANCE ASSESSMENT")
    print("-" * 60)
    
    target_reward = -150.0
    distance_to_target = abs(avg_reward - target_reward)
    
    print(f"Target Reward: {target_reward}")
    print(f"Current Reward: {avg_reward:.2f}")
    print(f"Distance to Target: {distance_to_target:.2f}")
    
    if avg_reward >= target_reward:
        print("\n✓ TARGET ACHIEVED!")
    else:
        print(f"\n✗ Need {distance_to_target:.2f} more improvement")
        
    # Identify issues
    print("\nISSUES IDENTIFIED:")
    if avg_util < 50:
        print("- Very low machine utilization (scheduling inefficiency)")
    if avg_late > 5:
        print("- Too many late jobs")
    if avg_completed < 25:
        print("- Not completing enough jobs")
        
    # Save results
    results_path = 'phase3/testing/small_rush_test_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': model_path,
            'n_episodes': n_episodes,
            'metrics': {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_late_jobs': avg_late,
                'avg_completed': avg_completed,
                'avg_utilization': avg_util,
                'avg_quality': avg_quality,
                'target_reward': target_reward,
                'distance_to_target': distance_to_target,
                'target_achieved': avg_reward >= target_reward
            },
            'raw_results': results
        }, f, indent=2)
        
    print(f"\nResults saved to: {results_path}")
    
    return avg_reward >= target_reward


if __name__ == '__main__':
    # Run from app_2 directory
    os.chdir('/Users/carrickcheah/Project/ppo/app_2')
    success = test_small_rush_model()
    
    if success:
        print("\nSmall rush stage is ready to proceed!")
    else:
        print("\nSmall rush stage needs more improvement before proceeding.")