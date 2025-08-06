#!/usr/bin/env python
"""
Quick training script for testing PPO scheduler with reduced timesteps.
Focuses on getting a working model quickly rather than optimal performance.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum_trainer import CurriculumTrainer, StageConfig, PPOConfig

def get_quick_stages():
    """Quick training stages - much faster completion."""
    return [
        StageConfig(
            name="toy_easy",
            data_path="data/10_jobs.json",
            n_timesteps=5000,  # Very quick
            success_threshold=0.5,
            min_reward=0,
            description="Learn basic sequencing"
        ),
        StageConfig(
            name="toy_normal", 
            data_path="data/20_jobs.json",
            n_timesteps=10000,  # Quick
            success_threshold=0.4,
            min_reward=0,
            description="Handle urgency"
        ),
        StageConfig(
            name="small",
            data_path="data/40_jobs.json",
            n_timesteps=15000,  # Moderate
            success_threshold=0.3,
            min_reward=0,
            description="Resource contention"
        ),
        # Skip larger stages for quick training
    ]

if __name__ == "__main__":
    print("Starting quick training run...")
    print("This will train on 3 stages only for faster completion")
    print("-" * 50)
    
    # Optimized config for speed
    config = PPOConfig(
        learning_rate=1e-3,  # Higher LR for faster learning
        batch_size=32,  # Smaller batch for faster updates
        n_steps=512,  # Fewer steps per rollout
        n_epochs=5,  # Fewer epochs per update
        device="mps"
    )
    
    # Create trainer with quick stages
    trainer = CurriculumTrainer(
        stages=get_quick_stages(),
        checkpoint_dir="checkpoints/quick",
        tensorboard_dir="tensorboard/quick",
        config=config
    )
    
    # Run training
    results = trainer.train_curriculum()
    
    print("\nQuick training complete!")
    print(f"Final model saved to: checkpoints/quick/curriculum_final.pth")
    print("\nYou can now test the model with:")
    print("  uv run python src/evaluation/test_model.py --model checkpoints/quick/curriculum_final.pth")