#!/usr/bin/env python3
"""
Run script for Phase 4 Extended Training

This script runs the extended training from the app directory.
"""

from src.training.train_phase4_extended import main

if __name__ == "__main__":
    print("Starting Phase 4 Extended Training (2M timesteps)...")
    print("This will take approximately 20 hours.")
    print("Target: Reduce makespan from 49.2h to <45h")
    print("-" * 50)
    
    main()