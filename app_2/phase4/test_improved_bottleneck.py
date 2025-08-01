"""
Test the improved bottleneck environment to validate fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environments.improved_bottleneck_env import ImprovedBottleneckEnvironment

def test_improved_environment():
    """Test the improved environment functionality."""
    env = ImprovedBottleneckEnvironment(verbose=True)
    
    print("\n=== IMPROVED ENVIRONMENT INFO ===")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Number of families: {len(env.families)}")
    print(f"Number of machines: {len(env.machines)}")
    print(f"Total tasks: {env.total_tasks}")
    print(f"Max steps: {env.max_steps}")
    print(f"Time increment: {env.time_increment} hours")
    
    # Reset and check initial state
    obs, info = env.reset()
    print(f"\n=== INITIAL STATE ===")
    print(f"Valid actions available: {len(info.get('valid_actions', env._get_valid_actions()))}")
    print(f"Initial time: {env.current_time}")
    print(f"Action mask shape: {info.get('action_mask', 'Not available').shape if hasattr(info.get('action_mask', 'Not available'), 'shape') else 'Not available'}")
    
    # Test valid actions only
    print(f"\n=== TESTING VALID ACTIONS ONLY ===")
    total_reward = 0
    valid_schedules = 0
    invalid_actions = 0
    no_actions = 0
    
    valid_actions = env._get_valid_actions()
    print(f"Available valid actions: {len(valid_actions)}")
    
    for i in range(50):
        # Select random valid action
        if valid_actions:
            action = valid_actions[np.random.randint(len(valid_actions))]
            action = np.array(action, dtype=int)
        else:
            # Fallback to no-action
            action = np.array([len(env.family_ids), len(env.machine_ids)], dtype=int)
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Current time: {env.current_time:.1f} hours ({env.current_time/24:.1f} days)")
        print(f"  Scheduled jobs: {len(env.scheduled_jobs)}")
        print(f"  Action type: {info.get('action_type', 'unknown')}")
        
        if info.get('action_type') == 'schedule':
            valid_schedules += 1
            print(f"  Scheduled: {info.get('scheduled_job')}")
            if 'progress_bonus' in info:
                print(f"  Progress bonus: {info['progress_bonus']:.1f}")
            if 'utilization_bonus' in info:
                print(f"  Utilization bonus: {info['utilization_bonus']:.1f}")
        elif info.get('action_type') == 'invalid':
            invalid_actions += 1
            print(f"  WARNING: Invalid action occurred!")
        elif info.get('action_type') == 'no_action':
            no_actions += 1
            
        # Update valid actions for next iteration
        valid_actions = env._get_valid_actions()
        print(f"  Valid actions remaining: {len(valid_actions)}")
        
        if done or truncated:
            print(f"\nEpisode ended!")
            break
    
    print(f"\n=== SUMMARY ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Valid schedules: {valid_schedules}")
    print(f"Invalid actions: {invalid_actions}")
    print(f"No actions: {no_actions}")
    print(f"Final time: {env.current_time:.1f} hours ({env.current_time/24:.1f} days)")
    print(f"Completion rate: {len(env.scheduled_jobs) / env.total_tasks * 100:.1f}%")
    print(f"Machine utilization: {env._calculate_utilization()*100:.1f}%")
    
    # Test action masking
    print(f"\n=== ACTION MASKING TEST ===")
    env.reset()
    mask = env.get_action_mask()
    print(f"Action mask shape: {mask.shape}")
    print(f"Valid action pairs: {np.sum(mask)}")
    print(f"Total possible pairs: {np.prod(mask.shape)}")
    print(f"Valid action percentage: {np.sum(mask)/np.prod(mask.shape)*100:.1f}%")
    
    # Test time progression
    print(f"\n=== TIME PROGRESSION TEST ===")
    env.reset()
    times = []
    for i in range(10):
        action = np.array([len(env.family_ids), len(env.machine_ids)])  # No-action
        obs, reward, done, truncated, info = env.step(action)
        times.append(env.current_time)
    print(f"Time after 10 no-action steps: {times}")
    print(f"Time increment per step: {times[1] - times[0] if len(times) > 1 else 'N/A'} hours")
    
    # Test reward improvements
    print(f"\n=== REWARD STRUCTURE TEST ===")
    env.reset()
    valid_actions = env._get_valid_actions()
    if valid_actions:
        # Test valid scheduling action
        action = np.array(valid_actions[0])
        obs, reward, done, truncated, info = env.step(action)
        print(f"Valid scheduling reward: {reward:.2f}")
        print(f"Reward breakdown: {info.get('reward_breakdown', 'Not available')}")
        
        # Test invalid action (should be prevented by our improvements)
        invalid_action = np.array([len(env.family_ids)-1, len(env.machine_ids)-1])  
        if not env._is_action_valid(invalid_action[0], invalid_action[1]):
            obs, reward, done, truncated, info = env.step(invalid_action)
            print(f"Invalid action penalty: {reward:.2f}")
    
    print(f"\n=== COMPARISON TO ORIGINAL ===")
    print("Improvements made:")
    print("✓ Time increment: 0.1h -> 1.0h (10x faster)")
    print("✓ Episode length: 300 -> 500 steps (better coverage)")
    print("✓ Invalid action penalty: -10 -> -50 (stronger deterrent)")
    print("✓ Valid schedule reward: 15+25 -> 50+100 (much higher)")
    print("✓ Added progress bonus: +30 (encourages action)")
    print("✓ Added utilization bonus: +20 (scaled by utilization)")
    print("✓ Pre-computed valid pairs for efficiency")
    print("✓ Action masking available")
    print("✓ Better evaluation with stochastic option")


def compare_environments():
    """Compare original vs improved environment performance."""
    print(f"\n{'='*70}")
    print("ENVIRONMENT COMPARISON")
    print(f"{'='*70}")
    
    # Test original environment
    try:
        from environments.small_bottleneck_env import SmallBottleneckEnvironment
        
        print("\n--- Original Environment ---")
        orig_env = SmallBottleneckEnvironment(verbose=False)
        orig_obs, orig_info = orig_env.reset()
        
        orig_rewards = []
        orig_completions = []
        
        for episode in range(5):
            obs, info = orig_env.reset()
            episode_reward = 0
            for step in range(50):
                action = orig_env.action_space.sample()
                obs, reward, done, truncated, info = orig_env.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            
            completion = len(orig_env.scheduled_jobs) / orig_env.total_tasks
            orig_rewards.append(episode_reward)
            orig_completions.append(completion)
        
        print(f"Original avg reward: {np.mean(orig_rewards):.2f}")
        print(f"Original avg completion: {np.mean(orig_completions)*100:.1f}%")
        
    except Exception as e:
        print(f"Could not test original environment: {e}")
    
    # Test improved environment  
    print("\n--- Improved Environment ---")
    imp_env = ImprovedBottleneckEnvironment(verbose=False)
    
    imp_rewards = []
    imp_completions = []
    
    for episode in range(5):
        obs, info = imp_env.reset()
        episode_reward = 0
        
        for step in range(50):
            # Use valid actions only
            valid_actions = imp_env._get_valid_actions()
            if valid_actions:
                action = valid_actions[np.random.randint(len(valid_actions))]
                action = np.array(action)
            else:
                action = np.array([len(imp_env.family_ids), len(imp_env.machine_ids)])
                
            obs, reward, done, truncated, info = imp_env.step(action)
            episode_reward += reward
            if done or truncated:
                break
        
        completion = len(imp_env.scheduled_jobs) / imp_env.total_tasks
        imp_rewards.append(episode_reward)
        imp_completions.append(completion)
    
    print(f"Improved avg reward: {np.mean(imp_rewards):.2f}")
    print(f"Improved avg completion: {np.mean(imp_completions)*100:.1f}%")
    
    # Calculate improvement
    try:
        reward_improvement = (np.mean(imp_rewards) - np.mean(orig_rewards)) / abs(np.mean(orig_rewards)) * 100
        completion_improvement = (np.mean(imp_completions) - np.mean(orig_completions)) / max(np.mean(orig_completions), 0.01) * 100
        
        print(f"\nImprovement:")
        print(f"Reward: {reward_improvement:+.1f}%")
        print(f"Completion: {completion_improvement:+.1f}%")
    except:
        print("\nCould not calculate improvement percentages")


if __name__ == "__main__":
    test_improved_environment()
    compare_environments()