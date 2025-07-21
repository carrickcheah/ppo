"""
Load REAL production data from MariaDB database.
This script MUST be run before any testing or training to ensure real data usage.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

from src.data_ingestion.ingest_data import ProductionDataIngester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_real_data(snapshot_data):
    """Validate that the data is real production data, not synthetic."""
    validations = {
        'has_real_job_prefixes': False,
        'has_real_machine_names': False,
        'has_database_timestamps': False,
        'has_real_lcd_dates': False
    }
    
    # Check job prefixes
    real_prefixes = ['JOAW', 'JOST', 'JOEX', 'JOCS', 'JOPT']
    families = snapshot_data.get('families', {})
    
    for family_id, family_data in families.items():
        # Check if job ID has real prefix
        if any(family_id.startswith(prefix) for prefix in real_prefixes):
            validations['has_real_job_prefixes'] = True
            
        # Check for database timestamps
        if family_data.get('created_date') and 'T' in family_data['created_date']:
            validations['has_database_timestamps'] = True
            
        # Check LCD dates are realistic
        if family_data.get('lcd_date'):
            validations['has_real_lcd_dates'] = True
            
    # Check machine names
    machines = snapshot_data.get('machines', [])
    real_machine_patterns = ['CM', 'CL', 'AD', 'MG', 'PP', 'ST', 'TP', 'WS']
    
    for machine in machines:
        machine_name = machine.get('machine_name', '')
        if any(pattern in machine_name for pattern in real_machine_patterns):
            validations['has_real_machine_names'] = True
            break
    
    # Validate all checks passed
    all_valid = all(validations.values())
    
    if not all_valid:
        failed_checks = [k for k, v in validations.items() if not v]
        raise ValueError(f"Data validation failed! Not real production data. Failed checks: {failed_checks}")
    
    logger.info("✓ Data validation passed - confirmed REAL production data")
    return True

def load_real_production_data():
    """Load real production data from database."""
    logger.info("=== Loading REAL Production Data ===")
    logger.info("Per CLAUDE.md requirements: ONLY real data allowed, NO synthetic data")
    
    # Check for existing snapshot
    snapshot_path = Path("data/real_production_snapshot.json")
    
    if snapshot_path.exists():
        # Check age of snapshot
        file_age = datetime.now() - datetime.fromtimestamp(snapshot_path.stat().st_mtime)
        logger.info(f"Found existing snapshot (age: {file_age})")
        
        if file_age.days > 1:
            logger.warning("Snapshot is older than 1 day, fetching fresh data...")
            fetch_new = True
        else:
            fetch_new = False
    else:
        logger.info("No snapshot found, fetching from database...")
        fetch_new = True
    
    if fetch_new:
        # Create data ingester
        ingester = ProductionDataIngester()
        
        # Test connection
        try:
            conn = ingester.connect()
            logger.info("✓ Database connection successful")
            conn.close()
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            logger.error("Cannot proceed without real data - synthetic data is FORBIDDEN")
            raise
        
        # Create snapshot
        try:
            output_file = ingester.create_snapshot(
                planning_horizon_days=30,
                output_file=str(snapshot_path),
                include_machines=True
            )
            logger.info(f"✓ Created new production snapshot: {output_file}")
        except Exception as e:
            logger.error(f"✗ Failed to create snapshot: {e}")
            raise
    
    # Load and validate the snapshot
    with open(snapshot_path, 'r') as f:
        snapshot_data = json.load(f)
    
    # Validate it's real data
    validate_real_data(snapshot_data)
    
    # Show summary
    metadata = snapshot_data.get('metadata', {})
    stats = snapshot_data.get('statistics', {})
    
    logger.info("\n=== Production Data Summary ===")
    logger.info(f"Created: {metadata.get('created_at', 'Unknown')}")
    logger.info(f"Database: {metadata.get('database', 'Unknown')}")
    logger.info(f"Total Families: {metadata.get('total_families', 0)}")
    logger.info(f"Total Tasks: {metadata.get('total_tasks', 0)}")
    logger.info(f"Total Machines: {metadata.get('total_machines', 0)}")
    logger.info(f"Urgent Jobs: {stats.get('urgent_families', 0)}")
    logger.info(f"Critical Jobs: {stats.get('critical_families', 0)}")
    
    # Sample some real job IDs to prove it's real data
    logger.info("\n=== Sample Real Job IDs ===")
    families = snapshot_data.get('families', {})
    for i, (family_id, family_data) in enumerate(list(families.items())[:5]):
        logger.info(f"- {family_id} (LCD: {family_data.get('lcd_date', 'N/A')})")
    
    return snapshot_data

def update_environment_to_use_real_data():
    """Update environment configuration to use real data."""
    logger.info("\n=== Updating Environment Configuration ===")
    
    # Path to real data
    real_data_path = Path("data/real_production_snapshot.json")
    
    if not real_data_path.exists():
        logger.error("Real production snapshot not found! Run this script first to fetch data.")
        raise FileNotFoundError("No real production data available")
    
    # Create a marker file to indicate real data mode
    marker_path = Path("data/.use_real_data_only")
    marker_path.parent.mkdir(exist_ok=True)
    
    with open(marker_path, 'w') as f:
        f.write(f"REAL DATA MODE ENFORCED\n")
        f.write(f"Snapshot: {real_data_path}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write("Any attempt to use synthetic data will raise an error\n")
    
    logger.info("✓ Environment configured to use REAL data only")
    logger.info(f"✓ Data source: {real_data_path}")

if __name__ == "__main__":
    try:
        # Load real data
        snapshot_data = load_real_production_data()
        
        # Configure environment
        update_environment_to_use_real_data()
        
        logger.info("\n✅ SUCCESS: Real production data loaded and validated")
        logger.info("You can now run tests and training with REAL data only")
        
    except Exception as e:
        logger.error(f"\n❌ FAILED: {e}")
        logger.error("Cannot proceed - synthetic data is FORBIDDEN per CLAUDE.md")
        sys.exit(1)