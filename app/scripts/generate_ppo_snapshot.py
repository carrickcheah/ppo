#!/usr/bin/env python3
"""
Generate PPO training snapshot from production database.
Fetches latest jobs and converts to PPO training format.
"""

import sys
import os
from pathlib import Path
import json
import logging
import argparse
from datetime import datetime

# Add paths for imports
app_path = Path(__file__).parent.parent
sys.path.insert(0, str(app_path))

from src.data_ingestion.database_to_ppo_adapter import DatabaseToPPOAdapter
from src.data_ingestion.mariadb_connection import load_ppo_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_snapshot(
    max_jobs: int = 1000,
    planning_horizon: int = 30,
    output_dir: str = None,
    use_mock: bool = False
) -> str:
    """
    Generate PPO training snapshot from production data.
    
    Args:
        max_jobs: Maximum number of jobs to fetch
        planning_horizon: Planning horizon in days
        output_dir: Output directory for snapshot
        
    Returns:
        Path to generated snapshot file
    """
    if output_dir is None:
        output_dir = app_path / 'data'
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize adapter
    adapter = DatabaseToPPOAdapter()
    
    # Load data
    if use_mock:
        logger.info(f"Loading mock data for testing...")
    else:
        logger.info(f"Loading up to {max_jobs} jobs from database...")
        logger.info(f"Planning horizon: {planning_horizon} days")
    
    try:
        jobs, machines, setup_times = load_ppo_data(
            use_mock=use_mock,
            max_jobs=max_jobs,
            planning_horizon_days=planning_horizon
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    if not jobs:
        logger.error("No jobs loaded from database")
        return None
        
    logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
    
    # Convert to PPO format
    logger.info("Converting to PPO format...")
    ppo_data = adapter.convert_to_ppo_format(jobs, machines, setup_times)
    
    # Add snapshot metadata
    ppo_data['snapshot_info'] = {
        'generated_at': datetime.now().isoformat(),
        'max_jobs_requested': max_jobs,
        'actual_jobs_loaded': len(jobs),
        'planning_horizon_days': planning_horizon,
        'family_count': len(ppo_data['families']),
        'machine_count': len(ppo_data['machines']),
        'total_tasks': sum(len(f['tasks']) for f in ppo_data['families'].values())
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ppo_production_snapshot_{timestamp}.json'
    filepath = output_dir / filename
    
    # Save snapshot
    logger.info(f"Saving snapshot to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(ppo_data, f, indent=2)
    
    # Also save as 'latest' for convenience
    latest_path = output_dir / 'ppo_production_snapshot_latest.json'
    with open(latest_path, 'w') as f:
        json.dump(ppo_data, f, indent=2)
    logger.info(f"Also saved as {latest_path}")
    
    # Print summary
    print("\n=== Snapshot Summary ===")
    print(f"Generated: {ppo_data['snapshot_info']['generated_at']}")
    print(f"Jobs loaded: {ppo_data['snapshot_info']['actual_jobs_loaded']}")
    print(f"Families created: {ppo_data['snapshot_info']['family_count']}")
    print(f"Total tasks: {ppo_data['snapshot_info']['total_tasks']}")
    print(f"Machines: {ppo_data['snapshot_info']['machine_count']}")
    
    # Show job urgency distribution
    urgent_families = sum(
        1 for f in ppo_data['families'].values() 
        if f.get('is_important', False)
    )
    print(f"Important families: {urgent_families} ({urgent_families/len(ppo_data['families'])*100:.1f}%)")
    
    # Show sample families
    print("\n=== Sample Families ===")
    sample_families = list(ppo_data['families'].values())[:5]
    for family in sample_families:
        lcd_days = family.get('lcd_days_remaining', 'N/A')
        material_days = family.get('material_days_ago', 'N/A')
        print(
            f"Family {family['family_id']}: "
            f"{len(family['tasks'])} tasks, "
            f"important={family['is_important']}, "
            f"LCD in {lcd_days} days, "
            f"material {material_days} days ago"
        )
    
    return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate PPO training snapshot from production database"
    )
    parser.add_argument(
        '--max-jobs', 
        type=int, 
        default=1000,
        help='Maximum number of jobs to fetch (default: 1000)'
    )
    parser.add_argument(
        '--horizon', 
        type=int, 
        default=30,
        help='Planning horizon in days (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for snapshot (default: app/data)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with only 100 jobs'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock data instead of database'
    )
    
    args = parser.parse_args()
    
    # Override for test mode
    if args.test:
        args.max_jobs = 100
        logger.info("Running in test mode with 100 jobs")
    
    try:
        filepath = generate_snapshot(
            max_jobs=args.max_jobs,
            planning_horizon=args.horizon,
            output_dir=args.output_dir,
            use_mock=args.mock
        )
        
        if filepath:
            print(f"\nSnapshot saved to: {filepath}")
            return 0
        else:
            print("\nFailed to generate snapshot")
            return 1
            
    except Exception as e:
        logger.error(f"Error generating snapshot: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())