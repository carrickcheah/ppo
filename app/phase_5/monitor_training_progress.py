#!/usr/bin/env python3
"""
Monitor Phase 5 training progress by checking saved checkpoints
"""

import os
import glob
import yaml
from datetime import datetime
from pathlib import Path

def check_training_progress():
    print("\n" + "="*60)
    print("Phase 5 Training Progress Monitor")
    print("="*60 + "\n")
    
    # Check for checkpoints
    checkpoint_dir = "models/multidiscrete/extended"
    checkpoint_pattern = f"{checkpoint_dir}/phase5_extended_*.zip"
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints:")
        for ckpt in checkpoints:
            # Extract step number from filename
            filename = os.path.basename(ckpt)
            if '_steps' in filename:
                steps = filename.split('_')[-2]
                print(f"  - {steps} steps: {ckpt}")
            else:
                print(f"  - {ckpt}")
        
        # Get latest checkpoint
        latest = checkpoints[-1]
        latest_steps = int(os.path.basename(latest).split('_')[-2])
        progress = latest_steps / 2000000 * 100
        print(f"\nLatest checkpoint: {latest_steps:,} steps ({progress:.1f}% complete)")
    else:
        print("No extended training checkpoints found yet")
        
    # Check for fixed training checkpoints as baseline
    fixed_dir = "models/multidiscrete/fixed"
    fixed_pattern = f"{fixed_dir}/phase5_fixed_*.zip"
    fixed_checkpoints = sorted(glob.glob(fixed_pattern))
    
    if fixed_checkpoints:
        print(f"\nFixed training checkpoints (baseline):")
        for ckpt in fixed_checkpoints:
            filename = os.path.basename(ckpt)
            if '_steps' in filename:
                steps = filename.split('_')[-2]
                print(f"  - {steps} steps")
    
    # Check for results file
    results_file = "results/phase5/extended_training_results.yaml"
    if os.path.exists(results_file):
        print(f"\n✅ Training completed! Results available at: {results_file}")
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
        print(f"  Final makespan: {results.get('final_makespan', 'N/A')} hours")
        print(f"  Jobs scheduled: {results.get('final_scheduled', 'N/A')}/411")
        print(f"  Training time: {results.get('training_time_minutes', 'N/A'):.1f} minutes")
    else:
        print("\nTraining still in progress or not yet started...")
        print("Expected completion: 30-40 minutes from start")
        
    # Check logs
    log_dir = "logs/phase5/extended"
    if os.path.exists(log_dir):
        log_files = sorted(glob.glob(f"{log_dir}/*"))
        if log_files:
            print(f"\nEvaluation logs found: {len(log_files)} files")
            latest_log = max(log_files, key=os.path.getmtime)
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_log))
            print(f"  Latest update: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_training_progress()