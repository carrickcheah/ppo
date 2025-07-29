"""
Quick test of all strategy environments
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4.environments import (
    SmallBalancedEnvironment,
    SmallRushEnvironment,
    SmallBottleneckEnvironment,
    SmallComplexEnvironment
)


def test_environment(env_class, env_name):
    """Test a single environment."""
    print(f"\nTesting {env_name}")
    print("-" * 40)
    
    try:
        # Create environment
        env = env_class(verbose=False)
        
        # Print basic info
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print(f"Total families: {len(env.families)}")
        print(f"Total machines: {len(env.machines)}")
        print(f"Total tasks: {env.total_tasks}")
        
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Valid actions: {len(info.get('valid_actions', []))}")
        
        # Test a few random steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                print(f"Episode ended at step {step+1}")
                break
        
        print(f"Total reward after 10 steps: {total_reward:.1f}")
        print(f"Jobs scheduled: {len(env.scheduled_jobs)}/{env.total_tasks}")
        
        # Get metrics if available
        if hasattr(env, 'get_metrics_summary'):
            metrics = env.get_metrics_summary()
            print("Metrics:", metrics)
        
        print("✓ Environment working correctly")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Test all environments."""
    print("TESTING PHASE 4 STRATEGY ENVIRONMENTS")
    print("=" * 60)
    
    environments = [
        (SmallBalancedEnvironment, "Small Balanced"),
        (SmallRushEnvironment, "Small Rush"),
        (SmallBottleneckEnvironment, "Small Bottleneck"),
        (SmallComplexEnvironment, "Small Complex")
    ]
    
    for env_class, env_name in environments:
        test_environment(env_class, env_name)
    
    print("\n" + "="*60)
    print("All environment tests completed!")


if __name__ == "__main__":
    main()