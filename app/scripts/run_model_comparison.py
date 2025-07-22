#!/usr/bin/env python3
"""
Run comprehensive model comparison and evaluation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.evaluation.model_evaluator import ModelEvaluator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Run model comparison"""
    print("="*80)
    print("PPO Production Scheduler - Model Comparison")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Define models to compare
    models_to_compare = []
    
    # Phase 4 model (if exists)
    phase4_path = "models/full_production/final_model.zip"
    if Path(phase4_path).exists():
        models_to_compare.append((phase4_path, "Phase 4 - Production", "full_production"))
        print(f"✓ Found Phase 4 model: {phase4_path}")
    
    # Phase 5 models (if exist)
    phase5_paths = [
        ("models/multidiscrete/correct_dims/phase5_320jobs_100000_steps.zip", "Phase 5 - 100k"),
        ("models/multidiscrete/exploration_continued/phase5_explore_300000_steps.zip", "Phase 5 - 300k (Best)"),
        ("models/multidiscrete/exploration_continued/phase5_explore_750000_steps.zip", "Phase 5 - 750k"),
    ]
    
    for path, name in phase5_paths:
        if Path(path).exists():
            models_to_compare.append((path, name, "hierarchical"))
            print(f"✓ Found Phase 5 model: {name}")
    
    if not models_to_compare:
        print("\nNo models found to compare!")
        print("Please ensure models are trained and saved in the expected locations.")
        return
    
    print(f"\nComparing {len(models_to_compare)} PPO models + 4 baseline algorithms")
    print("-"*80)
    
    # Run comparison
    try:
        evaluator.compare_models(models_to_compare)
        print("\nComparison complete! Check the generated report files.")
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()