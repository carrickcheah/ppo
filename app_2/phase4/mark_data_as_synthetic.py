#!/usr/bin/env python3
"""
Update Phase 4 data files to clearly mark them as SYNTHETIC.
This is a temporary solution until real database connection is available.
"""

import json
from pathlib import Path


def mark_data_as_synthetic():
    """Update all Phase 4 data files to clearly indicate they use synthetic data."""
    
    data_dir = Path(__file__).parent / "data"
    
    for data_file in data_dir.glob("*.json"):
        print(f"Updating {data_file.name}...")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Update description to clearly indicate synthetic data
        data['description'] = f"SYNTHETIC DATA - {data.get('description', '')}"
        
        # Add warning to metrics
        if 'metrics' not in data:
            data['metrics'] = {}
        
        data['metrics']['WARNING'] = "This uses SYNTHETIC data, not real production data from MariaDB"
        data['metrics']['data_source'] = "SYNTHETIC - Generated for testing (violates CLAUDE.md requirements)"
        
        # Save updated file
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  - Marked as SYNTHETIC")
    
    print("\n" + "="*70)
    print("WARNING: Phase 4 is currently using SYNTHETIC data")
    print("This violates CLAUDE.md requirement to use ONLY real production data")
    print("To fix: Connect to MariaDB and run fetch_real_production_data.py")
    print("="*70)


if __name__ == "__main__":
    mark_data_as_synthetic()