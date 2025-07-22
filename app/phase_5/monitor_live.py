#!/usr/bin/env python3
"""
Live monitoring of Phase 5 training progress
"""

import time
import subprocess
import os
from datetime import datetime

def get_latest_timesteps():
    """Extract latest timestep count from log"""
    try:
        result = subprocess.run(
            ["tail", "-50", "phase_5/training_log.txt"],
            capture_output=True, text=True
        )
        lines = result.stdout.split('\n')
        for line in reversed(lines):
            if 'total_timesteps' in line:
                # Extract number from line like "|    total_timesteps      | 149152      |"
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        return int(parts[2].strip())
                    except:
                        pass
        return 0
    except:
        return 0

def check_checkpoints():
    """Check for saved checkpoints"""
    checkpoints = []
    extended_dir = "models/multidiscrete/extended"
    if os.path.exists(extended_dir):
        for f in os.listdir(extended_dir):
            if f.endswith('.zip') and 'phase5_extended' in f:
                checkpoints.append(f)
    return sorted(checkpoints)

def monitor():
    print("Phase 5 Training Monitor - Press Ctrl+C to stop")
    print("="*60)
    
    start_time = datetime.now()
    last_timesteps = 100000  # Starting from 100k checkpoint
    
    while True:
        current_timesteps = get_latest_timesteps()
        if current_timesteps == 0:
            current_timesteps = last_timesteps
            
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = current_timesteps / 2000000 * 100
        
        # Calculate ETA
        if current_timesteps > 100000:
            rate = (current_timesteps - 100000) / elapsed
            remaining = 2000000 - current_timesteps
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 40  # Initial estimate
            
        # Check for checkpoints
        checkpoints = check_checkpoints()
        
        # Clear line and print status
        print(f"\rTimesteps: {current_timesteps:,}/2,000,000 ({progress:.1f}%) | "
              f"ETA: {eta_minutes:.0f}m | Checkpoints: {len(checkpoints)}", end='', flush=True)
        
        if current_timesteps >= 2000000:
            print("\n\nTraining completed!")
            break
            
        last_timesteps = current_timesteps
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")