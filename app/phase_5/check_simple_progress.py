#!/usr/bin/env python3
"""
Check simplified training progress
"""

import os
import glob
import subprocess

def check_progress():
    print("\n" + "="*60)
    print("Phase 5 Simplified Training Progress")
    print("="*60 + "\n")
    
    # Check if process is running
    try:
        result = subprocess.run(["pgrep", "-f", "train_phase5_simple"], capture_output=True)
        if result.returncode == 0:
            print("✓ Training is currently running")
        else:
            print("✗ Training has completed or stopped")
    except:
        pass
    
    # Check for checkpoints
    checkpoints = sorted(glob.glob("models/multidiscrete/simple/*.zip"))
    if checkpoints:
        print(f"\nCheckpoints saved: {len(checkpoints)}")
        for ckpt in checkpoints:
            print(f"  - {os.path.basename(ckpt)}")
    
    # Check log file
    log_file = "phase_5/simple_training_log.txt"
    if os.path.exists(log_file):
        # Get last few lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Look for completion markers
            for line in lines[-20:]:
                if "Final:" in line or "Makespan:" in line or "Testing model" in line:
                    print(f"\n{line.strip()}")
                elif "total_timesteps" in line and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        timesteps = parts[2].strip()
                        print(f"\nLatest timesteps: {timesteps}")
        except:
            pass
    
    # Check if 500k model exists
    if os.path.exists("models/multidiscrete/simple/model_500k.zip"):
        print("\n✅ 500k model successfully saved!")
        print("Ready for evaluation")

if __name__ == "__main__":
    check_progress()