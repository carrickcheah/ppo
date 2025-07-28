"""
Train only toy_normal with phased approach
Quick test to see if the approach works
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_toys_achievable import train_with_phases
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Training toy_normal with phased approach...")
    logger.info("Goal: Achieve high completion rate first, then optimize")
    
    # Train with fewer timesteps for quick test
    results = train_with_phases('toy_normal', max_timesteps=100000)
    
    print("\n" + "="*60)
    print("TOY_NORMAL TRAINING RESULTS")
    print("="*60)
    print(f"Completion rate: {results['final_rate']:.1%}")
    print(f"Average reward: {results['average_reward']:.1f}")
    print(f"Phase switched: {results['phase_switched']}")
    
    if results['final_rate'] >= 0.9:
        print("\n✓ SUCCESS! Achieved high completion rate!")
        print("The phased approach works - now we can train all stages.")
    else:
        print(f"\n⚠ Still needs work: {results['final_rate']:.1%} < 90%")
        print("May need more timesteps or further reward adjustments.")