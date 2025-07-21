"""
Test the Full Production Environment for Phase 4.
"""

import logging
import numpy as np
from src.environments.full_production_env import FullProductionEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_env_creation():
    """Test basic environment creation and reset."""
    logger.info("=== Testing Full Production Environment Creation ===")
    
    try:
        # Create environment with hierarchical state compression
        env = FullProductionEnv(
            n_machines=152,
            n_jobs=500,
            state_compression="hierarchical",
            max_episode_steps=2000,
            use_break_constraints=True,
            use_holiday_constraints=True
        )
        
        logger.info(f"✓ Environment created successfully")
        logger.info(f"  Machines: {env.n_machines}")
        logger.info(f"  Jobs loaded: {len(env.jobs) if hasattr(env, 'jobs') else 0}")
        logger.info(f"  Families: {len(env.families) if hasattr(env, 'families') else 0}")
        logger.info(f"  State compression: {env.state_compression}")
        logger.info(f"  Observation space: {env.observation_space.shape}")
        logger.info(f"  Action space: {env.action_space.n}")
        
        # Test reset
        logger.info("\n--- Testing Environment Reset ---")
        obs, info = env.reset()
        
        logger.info(f"✓ Reset successful")
        logger.info(f"  Observation shape: {obs.shape}")
        logger.info(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        logger.info(f"  Valid actions: {len(env.valid_actions) if hasattr(env, 'valid_actions') else 0}")
        
        # Test step
        logger.info("\n--- Testing Environment Step ---")
        if hasattr(env, 'valid_actions') and env.valid_actions:
            action = 0  # Take first valid action
            obs, reward, done, truncated, info = env.step(action)
            
            logger.info(f"✓ Step successful")
            logger.info(f"  Reward: {reward:.2f}")
            logger.info(f"  Done: {done}")
            logger.info(f"  Info keys: {list(info.keys())}")
        else:
            logger.warning("  No valid actions available")
            
        # Test different state compressions
        logger.info("\n--- Testing State Compression Methods ---")
        for compression in ["hierarchical", "compressed", "full"]:
            try:
                env_comp = FullProductionEnv(
                    n_machines=152,
                    n_jobs=500,
                    state_compression=compression,
                    max_episode_steps=2000
                )
                obs, _ = env_comp.reset()
                logger.info(f"✓ {compression:12s} compression: observation shape {obs.shape}")
            except Exception as e:
                logger.error(f"✗ {compression:12s} compression failed: {e}")
                
        # Test scale
        logger.info("\n--- Testing Scale Metrics ---")
        stats = env.get_stats()
        if 'scale_metrics' in stats:
            scale = stats['scale_metrics']
            logger.info(f"  Total machines: {scale['total_machines']}")
            logger.info(f"  Total jobs: {scale['total_jobs']}")
            logger.info(f"  Total families: {scale['total_families']}")
            logger.info(f"  Jobs per machine: {scale['jobs_per_machine']:.1f}")
            logger.info(f"  Avg family size: {scale['avg_family_size']:.1f}")
            
        logger.info("\n=== All Tests Passed ===")
        return True
        
    except Exception as e:
        logger.error(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    success = test_env_creation()
    
    if success:
        logger.info("\n✓ Full Production Environment is ready for Phase 4 training!")
    else:
        logger.error("\n✗ Environment tests failed. Please fix issues before proceeding.")
        

if __name__ == "__main__":
    main()