"""
Test Stage Performance

Simple test to evaluate any stage's performance.
"""

import os
import sys
import yaml
import json
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from phase3.environments.curriculum_env import CurriculumSchedulingEnv


def test_stage(stage_name: str, n_episodes: int = 10):
    """Test a specific stage's performance."""
    print("=" * 80)
    print(f"TESTING STAGE: {stage_name}")
    print("=" * 80)
    print()
    
    # Load curriculum config
    with open('configs/phase3_curriculum_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    if stage_name not in config:
        print(f"Error: Stage '{stage_name}' not found in config")
        return None
        
    stage_config = config[stage_name]
    
    # Determine snapshot path
    snapshot_path = None
    data_source = stage_config['data_source']
    
    snapshot_mapping = {
        'snapshot_normal': 'phase3/snapshots/snapshot_normal.json',
        'snapshot_rush': 'phase3/snapshots/snapshot_rush.json',
        'edge_same_machine': 'phase3/snapshots/edge_case_same_machine.json',
    }
    
    if 'rush' in data_source:
        snapshot_path = snapshot_mapping['snapshot_rush']
    elif 'bottleneck' in data_source or 'same_machine' in data_source:
        snapshot_path = snapshot_mapping['edge_same_machine']
    elif data_source != 'synthetic':
        snapshot_path = snapshot_mapping['snapshot_normal']
        
    # Create fresh environment
    env = CurriculumSchedulingEnv(
        stage_config=stage_config,
        snapshot_path=snapshot_path,
        reward_profile='balanced',
        seed=42
    )
    
    # Create vec env
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Load model
    model_path = f'phase3/checkpoints/{stage_name}_final.zip'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
        
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=vec_env, device='cpu')
    
    # Disable training mode
    vec_env.training = False
    vec_env.norm_reward = False
    
    # Test metrics
    results = []
    
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
        metrics = {'episode': episode + 1, 'reward': episode_reward, 'steps': steps}
        
        if 'episode_metrics' in info[0]:
            ep_metrics = info[0]['episode_metrics']
            metrics.update({
                'completed': ep_metrics.get('jobs_completed', 0),
                'late': ep_metrics.get('jobs_late', 0),
                'utilization': ep_metrics.get('machine_utilization', 0) * 100
            })
            
        results.append(metrics)
        
        print(f"Episode {episode+1:2d} | "
              f"Reward: {episode_reward:6.1f} | "
              f"Steps: {steps:3d} | "
              f"Completed: {metrics.get('completed', 'N/A'):>3} | "
              f"Late: {metrics.get('late', 'N/A'):>3} | "
              f"Util: {metrics.get('utilization', 0):5.1f}%")
    
    # Calculate summary
    avg_reward = np.mean([r['reward'] for r in results])
    std_reward = np.std([r['reward'] for r in results])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    if 'completed' in results[0]:
        avg_completed = np.mean([r['completed'] for r in results])
        avg_late = np.mean([r['late'] for r in results])
        avg_util = np.mean([r['utilization'] for r in results])
        
        print(f"Average Completed: {avg_completed:.1f} / {stage_config['jobs']}")
        print(f"Average Late: {avg_late:.1f}")
        print(f"Average Utilization: {avg_util:.1f}%")
    
    # Save results
    result_data = {
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        'n_episodes': n_episodes,
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'episodes': [{k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()} for r in results]
    }
    
    result_path = f'phase3/testing/{stage_name}_test_results.json'
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
        
    print(f"\nResults saved to: {result_path}")
    
    return avg_reward


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test stage performance')
    parser.add_argument('--stage', type=str, required=True, help='Stage to test')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Change to app_2 directory
    os.chdir('/Users/carrickcheah/Project/ppo/app_2')
    
    avg_reward = test_stage(args.stage, args.episodes)
    
    if avg_reward is not None:
        print(f"\nFinal average reward: {avg_reward:.2f}")


if __name__ == '__main__':
    main()