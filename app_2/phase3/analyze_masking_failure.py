"""
Analyze why action masking isn't improving performance
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed
from phase3.environments.action_masked_env import ToyStageActionMasker


def analyze_action_space():
    """Analyze the action space with and without masking."""
    
    # Original environment
    env_orig = CurriculumEnvironmentTrulyFixed('toy_normal', verbose=False)
    print("Original Environment:")
    print(f"Action space: {env_orig.action_space}")
    print(f"Total actions: {env_orig.action_space.nvec[0]} x {env_orig.action_space.nvec[1]} = {np.prod(env_orig.action_space.nvec)}")
    
    # Masked environment
    env_masked = ToyStageActionMasker(env_orig)
    print(f"\nMasked Environment:")
    print(f"Action space: {env_masked.action_space}")
    print(f"Total actions: {env_masked.action_space.n}")
    
    # Test valid actions
    obs, _ = env_orig.reset()
    valid_actions = env_orig._get_valid_actions()
    print(f"\nInitial state:")
    print(f"Valid actions: {len(valid_actions)}")
    print(f"Valid percentage: {len(valid_actions) / np.prod(env_orig.action_space.nvec) * 100:.1f}%")
    
    # Show some valid actions
    print(f"\nFirst 10 valid actions:")
    for i, (job, machine) in enumerate(valid_actions[:10]):
        print(f"  Action {i}: Job {job}, Machine {machine}")
    
    # Check if no-action is included
    no_action = (env_orig.action_space.nvec[0] - 1, env_orig.action_space.nvec[1] - 1)
    if no_action in valid_actions:
        print(f"\n✓ No-action {no_action} is included")
    else:
        print(f"\n✗ No-action {no_action} is NOT included!")
    
    # Test action mask
    mask = env_masked._get_action_mask(env_masked)
    print(f"\nAction mask:")
    print(f"Shape: {mask.shape}")
    print(f"Valid actions in mask: {np.sum(mask)}")
    print(f"Mask percentage: {np.sum(mask) / len(mask) * 100:.1f}%")
    
    # Compare flattened indices
    print(f"\nFlattened action mapping:")
    n_jobs = env_orig.action_space.nvec[0]
    n_machines = env_orig.action_space.nvec[1]
    
    for job, machine in valid_actions[:5]:
        flat_idx = job * n_machines + machine
        print(f"  ({job}, {machine}) -> {flat_idx}")
        if flat_idx < len(mask):
            print(f"    Mask[{flat_idx}] = {mask[flat_idx]}")
    
    return env_orig, env_masked


def test_reward_structure():
    """Test if rewards are working correctly."""
    env = CurriculumEnvironmentTrulyFixed('toy_normal', verbose=False)
    
    print("\nReward Structure Analysis:")
    print("-" * 40)
    
    obs, _ = env.reset()
    
    # Try scheduling first available job
    valid_actions = env._get_valid_actions()
    if len(valid_actions) > 1:  # Exclude no-action
        action = valid_actions[0]
        print(f"\nTrying to schedule: Job {action[0]}, Machine {action[1]}")
        
        obs, reward, done, truncated, info = env.step(np.array(action))
        
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        
        if 'reward_breakdown' in info:
            print("\nReward breakdown:")
            for key, value in info['reward_breakdown'].items():
                print(f"  {key}: {value}")
    
    # Check toy_normal specific issues
    print("\n\nChecking toy_normal deadline pressure:")
    for fid, family in env.families.items():
        lcd_days = family.get('lcd_days_remaining', 0)
        total_hours = sum(task['processing_time'] for task in family['tasks'])
        print(f"\nFamily {fid}:")
        print(f"  Total processing: {total_hours:.1f} hours")
        print(f"  Deadline: {lcd_days * 24:.1f} hours")
        print(f"  Feasible: {'Yes' if total_hours <= lcd_days * 24 else 'No'}")


if __name__ == "__main__":
    print("ANALYZING ACTION MASKING FAILURE")
    print("=" * 60)
    
    env_orig, env_masked = analyze_action_space()
    test_reward_structure()
    
    print("\n\nCONCLUSIONS:")
    print("-" * 40)
    print("Possible issues:")
    print("1. The flattened action space (17x7=119) might be harder to learn than MultiDiscrete([8,6])")
    print("2. The model might be converging to always selecting no-action")
    print("3. The reward structure still penalizes late jobs too heavily")
    print("4. Action masking helps exploration but doesn't solve the fundamental reward problem")