"""
Run Phase 4: Full Production Scale Training Pipeline (Automated)
This version automatically continues without prompting.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

# Import the main function
from run_phase4_full_production import run_phase4_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Monkey patch input to always return 'y'
original_input = input
def auto_yes_input(prompt):
    logger.info(f"Auto-responding 'y' to: {prompt}")
    return 'y'

# Replace input function
import builtins
builtins.input = auto_yes_input

if __name__ == "__main__":
    try:
        run_phase4_pipeline()
    finally:
        # Restore original input
        builtins.input = original_input