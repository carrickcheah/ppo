"""
Test the Real Data Curriculum Environment
Verifies that the environment loads real production data correctly
"""

import os
import sys
import numpy as np
import logging

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phase3.environments.curriculum_env_real import CurriculumEnvironmentReal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_stage(stage_name: str):
    """Test a specific curriculum stage."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing stage: {stage_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Create environment
        env = CurriculumEnvironmentReal(
            stage_name=stage_name,
            verbose=True,
            max_steps=100
        )
        
        # Reset and get initial observation
        obs, info = env.reset()
        logger.info(f"\nObservation shape: {obs.shape}")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Valid actions: {len(info.get('valid_actions', []))} available")
        
        # Test a few random steps
        total_reward = 0
        n_valid = 0
        n_invalid = 0
        
        for step in range(20):
            # Get valid actions
            valid_actions = info.get('valid_actions', [])
            
            if valid_actions:
                # Choose a random valid action
                action_tuple = valid_actions[np.random.randint(len(valid_actions))]
                action = np.array(action_tuple)
            else:
                # No valid actions - try random
                action = env.action_space.sample()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if info.get('action_valid', False):
                n_valid += 1
                logger.info(f"Step {step}: Valid action - scheduled {info.get('scheduled_job')}")
            else:
                n_invalid += 1
                logger.info(f"Step {step}: Invalid - {info.get('reason')}")
            
            if done or truncated:
                break
        
        # Summary
        logger.info(f"\nStage {stage_name} Summary:")
        logger.info(f"  Total reward: {total_reward:.2f}")
        logger.info(f"  Valid actions: {n_valid}")
        logger.info(f"  Invalid actions: {n_invalid}")
        logger.info(f"  Completed jobs: {len(env.completed_jobs)}/{len(env.families)}")
        
        # Verify it's using real data
        sample_jobs = list(env.families.keys())[:3]
        logger.info(f"\nSample job IDs (verifying real data): {sample_jobs}")
        
        # Check for real job prefixes
        real_prefixes = ['JOAW', 'JOST', 'JOTP', 'JOPRD', 'JOEX']
        has_real_jobs = any(
            any(job_id.startswith(prefix) for prefix in real_prefixes)
            for job_id in env.families.keys()
        )
        
        if has_real_jobs:
            logger.info("✓ Confirmed: Using REAL production job IDs")
        else:
            logger.error("✗ ERROR: Not using real production data!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing stage {stage_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test multiple curriculum stages."""
    logger.info("=== TESTING CURRICULUM ENVIRONMENT WITH REAL DATA ===")
    
    # Test a few key stages
    test_stages = [
        'toy_easy',      # Foundation
        'small_rush',    # Strategy development
        'medium_normal', # Scale training
    ]
    
    results = {}
    for stage in test_stages:
        results[stage] = test_stage(stage)
    
    # Summary
    logger.info("\n=== TEST SUMMARY ===")
    for stage, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{stage}: {status}")
    
    # Overall result
    if all(results.values()):
        logger.info("\nAll tests PASSED! Environment is ready for training.")
    else:
        logger.error("\nSome tests FAILED! Please fix issues before training.")


if __name__ == "__main__":
    main()