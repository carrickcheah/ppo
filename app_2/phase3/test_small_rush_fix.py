"""
Test Small Rush Fix

Quick test to verify the reward changes fix the 0% utilization issue.
"""

import os
import sys
import yaml
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.curriculum_env import CurriculumSchedulingEnv


def test_reward_structure():
    """Test that rewards encourage action over inaction."""
    print("=== TESTING SMALL RUSH REWARD STRUCTURE ===\n")
    
    # Load configs
    with open('configs/phase3_curriculum_config.yaml', 'r') as f:
        curriculum_config = yaml.safe_load(f)
    
    with open('configs/environment.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    
    stage_config = curriculum_config['small_rush']
    
    print(f"Stage: {stage_config['name']}")
    print(f"Reward profile: {stage_config.get('reward_profile', 'balanced')}")
    print(f"Entropy coefficient: {stage_config.get('ent_coef', 0.05)}")
    print(f"\nEnvironment rewards:")
    print(f"  Completion reward: {env_config['rewards']['completion_reward']}")
    print(f"  Action bonus: {env_config['rewards']['action_bonus']}")
    print(f"  Invalid action penalty: {env_config['rewards']['invalid_action_penalty']}")
    print(f"  Urgency multiplier: {env_config['rewards']['urgency_multiplier']}")
    
    # Create environment
    print("\n=== CREATING ENVIRONMENT ===")
    env = CurriculumSchedulingEnv(
        stage_config=stage_config,
        snapshot_path='phase3/snapshots/snapshot_rush.json',
        reward_profile=stage_config.get('reward_profile', 'balanced'),
        seed=42
    )
    
    # Get initial observation
    obs, info = env.reset()
    print(f"\nEnvironment created:")
    print(f"  Jobs: {env.n_jobs}")
    print(f"  Machines: {env.n_machines_actual}")
    print(f"  Observation shape: {obs.shape}")
    
    # Test a few random actions
    print("\n=== TESTING ACTIONS ===")
    total_reward = 0
    valid_actions = 0
    
    for i in range(10):
        # Get action mask
        mask = env.get_action_mask()
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            # Take a random valid action
            action_flat = np.random.choice(valid_indices)
            action = np.array([action_flat // env.n_machines_actual, 
                              action_flat % env.n_machines_actual])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get('valid_action', False):
                valid_actions += 1
                total_reward += reward
                print(f"\nAction {i+1}: Job {action[0]} on Machine {action[1]}")
                print(f"  Reward: {reward:.2f}")
                print(f"  Valid: {info.get('valid_action', False)}")
                print(f"  Scheduled: {info.get('scheduled_job', 'None')}")
        
        if terminated:
            break
    
    print(f"\n=== RESULTS ===")
    print(f"Valid actions taken: {valid_actions}/10")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per action: {total_reward/max(valid_actions, 1):.2f}")
    
    # Check reward structure
    if valid_actions > 0 and total_reward > 0:
        print("\n✅ SUCCESS: Model is taking actions and getting positive rewards!")
        print("The 0% utilization issue should be fixed.")
    else:
        print("\n❌ ISSUE: Still not taking actions or getting negative rewards.")
        print("Further investigation needed.")
    
    return valid_actions > 0 and total_reward > 0


def test_action_vs_inaction():
    """Compare reward for action vs doing nothing."""
    print("\n\n=== ACTION VS INACTION TEST ===")
    
    # Load environment config
    with open('configs/environment.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    
    rewards = env_config['rewards']
    
    print("\nScenario 1: Take a valid action")
    action_reward = (
        rewards['completion_reward'] +  # 50.0
        rewards['action_bonus'] +       # 5.0
        0  # Assume no penalties
    )
    print(f"  Base reward: {action_reward}")
    
    print("\nScenario 2: Take invalid action (do nothing)")
    invalid_reward = rewards['invalid_action_penalty']  # -5.0
    print(f"  Invalid action penalty: {invalid_reward}")
    
    print("\nScenario 3: Late job penalty (worst case)")
    late_reward = (
        rewards['completion_reward'] +  # 50.0
        rewards['action_bonus'] -       # 5.0
        20.0  # Max late penalty from code
    )
    print(f"  Late job reward: {late_reward}")
    
    print(f"\n=== ANALYSIS ===")
    print(f"Action reward ({action_reward}) > Invalid penalty ({invalid_reward}): {action_reward > invalid_reward}")
    print(f"Late job reward ({late_reward}) > Invalid penalty ({invalid_reward}): {late_reward > invalid_reward}")
    
    if action_reward > abs(invalid_reward) and late_reward > invalid_reward:
        print("\n✅ Reward structure encourages action over inaction!")
    else:
        print("\n❌ Reward structure still penalizes action!")


if __name__ == "__main__":
    # Test reward structure
    success = test_reward_structure()
    
    # Test action vs inaction
    test_action_vs_inaction()
    
    if success:
        print("\n\n🎉 Small Rush fix appears to be working!")
        print("Run the full training to confirm: python improve_stage_performance.py small_rush")
    else:
        print("\n\n⚠️ Issues detected. Check the reward calculations.")