#!/usr/bin/env python3
"""
Quick test script to verify the environment setup.
Run this to make sure everything is working before starting model development.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the environment test."""
    print("Testing PPO Scheduling Environment Setup...")
    print("-" * 50)
    
    # Run the test
    test_file = Path(__file__).parent / "tests" / "test_environment.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print("Test failed!")
        print(e.stdout)
        print(e.stderr)
        return 1
        
    print("\nSetup complete! The environment is ready for Phase 2 (building the PPO model).")
    return 0

if __name__ == "__main__":
    sys.exit(main())