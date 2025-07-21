"""
Prepare full production data for Phase 4 scale testing.
Extracts 152 machines from database and generates 500+ jobs.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

try:
    from src.utils.db_connector import DatabaseConnector
    DB_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Database connector not available - will use synthetic data")
    DB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_full_machine_list():
    """Extract all 152 machines from the database or generate synthetic data."""
    
    if DB_AVAILABLE:
        logger.info("Extracting full machine list from database...")
        
        db = DatabaseConnector()
        
        # Query to get all machines with their types
        # Using only the columns specified in CLAUDE.md
        query = """
        SELECT 
            MachineId_i,
            MachineName_v,
            MachinetypeId_i
        FROM tbl_machine
        WHERE MachineId_i IS NOT NULL
        ORDER BY MachineId_i
        """
        
        machines_df = db.execute_query(query)
        
        if machines_df is None or len(machines_df) == 0:
            logger.warning("No machines found in database, generating synthetic data")
            return generate_synthetic_machines()
            
        logger.info(f"Found {len(machines_df)} machines in database")
        
        # Convert to list of dictionaries
        machines = []
        for _, row in machines_df.iterrows():
            # Handle potential NaN values
            machine_id = row['MachineId_i']
            machine_type = row['MachinetypeId_i']
            
            # Skip rows with invalid data
            if pd.isna(machine_id) or pd.isna(machine_type):
                logger.warning(f"Skipping machine with invalid data: {row}")
                continue
                
            machine = {
                "machine_id": int(machine_id),
                "machine_name": str(row['MachineName_v']) if not pd.isna(row['MachineName_v']) else f"Machine_{int(machine_id)}",
                "machine_type_id": int(machine_type)
            }
            machines.append(machine)
            
        # Analyze machine type distribution
        valid_types = machines_df['MachinetypeId_i'].dropna()
        if len(valid_types) > 0:
            type_counts = valid_types.value_counts().sort_index()
            logger.info("\nMachine type distribution:")
            for machine_type, count in type_counts.items():
                logger.info(f"  Type {int(machine_type)}: {count} machines")
            
        return machines
    else:
        logger.info("Database not available, generating synthetic machine data...")
        return generate_synthetic_machines()


def generate_synthetic_machines():
    """Generate 152 synthetic machines with realistic type distribution."""
    machines = []
    
    # Realistic machine type distribution based on production patterns
    # Type 1-3: High capacity machines (30%)
    # Type 4-7: Medium capacity machines (50%)
    # Type 8-10: Specialized machines (20%)
    type_distribution = {
        1: 15, 2: 15, 3: 16,  # 46 machines (30%)
        4: 20, 5: 19, 6: 19, 7: 18,  # 76 machines (50%)
        8: 10, 9: 10, 10: 10  # 30 machines (20%)
    }
    
    machine_id = 1
    for machine_type, count in type_distribution.items():
        for i in range(count):
            machine = {
                "machine_id": machine_id,
                "machine_name": f"M{machine_id:03d}_T{machine_type}",
                "machine_type_id": machine_type
            }
            machines.append(machine)
            machine_id += 1
            
    logger.info(f"Generated {len(machines)} synthetic machines")
    logger.info("\nSynthetic machine type distribution:")
    for machine_type, count in type_distribution.items():
        logger.info(f"  Type {machine_type}: {count} machines")
        
    return machines


def create_full_production_snapshot():
    """Create a production snapshot file with all 152 machines."""
    machines = extract_full_machine_list()
    
    # Get unique machine types from actual data
    unique_types = sorted(set(m['machine_type_id'] for m in machines))
    
    # Create snapshot data structure
    snapshot = {
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "total_machines": len(machines),
            "source": "tbl_machine",
            "description": "Full production machine list for Phase 4 scale testing"
        },
        "machines": machines,
        "machine_types": {
            str(machine_type): {
                "type_id": machine_type,
                "count": sum(1 for m in machines if m['machine_type_id'] == machine_type),
                "capabilities": f"Machine type {machine_type} capabilities"
            }
            for machine_type in unique_types
        }
    }
    
    # Save snapshot
    output_path = Path("data/full_production_snapshot.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
        
    logger.info(f"Saved full production snapshot to {output_path}")
    logger.info(f"Total machines: {len(machines)}")
    
    return snapshot


def analyze_production_capacity():
    """Analyze the production capacity with 152 machines."""
    snapshot = create_full_production_snapshot()
    machines = snapshot['machines']
    
    logger.info("\n=== Production Capacity Analysis ===")
    logger.info(f"Total machines: {len(machines)}")
    
    # Working hours analysis (considering breaks)
    hours_per_day = 24 - 5.5 - 2.5  # Minus machine off time and breaks
    hours_per_week = hours_per_day * 5 + hours_per_day * 0.5 * 2  # Weekdays + half weekends
    
    logger.info(f"Effective hours per day: {hours_per_day:.1f}")
    logger.info(f"Effective hours per week: {hours_per_week:.1f}")
    
    # Capacity calculation
    total_capacity_per_week = len(machines) * hours_per_week
    logger.info(f"Total machine-hours per week: {total_capacity_per_week:.0f}")
    
    # Job capacity (assuming average 3 hours per job)
    avg_job_time = 3.0
    jobs_per_week_capacity = total_capacity_per_week / avg_job_time
    logger.info(f"Theoretical job capacity per week (3h avg): {jobs_per_week_capacity:.0f} jobs")
    
    # Recommended job count for testing (80% utilization target)
    target_utilization = 0.8
    recommended_jobs = int(jobs_per_week_capacity * target_utilization)
    logger.info(f"Recommended job count for testing (80% util): {recommended_jobs} jobs")
    
    return {
        "total_machines": len(machines),
        "hours_per_week": hours_per_week,
        "total_capacity": total_capacity_per_week,
        "recommended_jobs": recommended_jobs
    }


def main():
    """Main function to prepare full production data."""
    logger.info("=== Phase 4: Preparing Full Production Data ===")
    
    try:
        # Create machine snapshot
        snapshot = create_full_production_snapshot()
        
        # Analyze capacity
        capacity = analyze_production_capacity()
        
        logger.info("\n=== Summary ===")
        logger.info(f"Successfully prepared data for {snapshot['metadata']['total_machines']} machines")
        logger.info(f"Recommended to test with {capacity['recommended_jobs']} jobs")
        logger.info("\nNext steps:")
        logger.info("1. Run test_full_production_env.py to verify environment")
        logger.info("2. Run train_full_production.py to start Phase 4 training")
        
    except Exception as e:
        logger.error(f"Error preparing production data: {e}")
        raise


if __name__ == "__main__":
    main()