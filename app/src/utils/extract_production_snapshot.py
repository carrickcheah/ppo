"""
Extract production data snapshot from database for training.
Creates a static JSON file with machines, jobs, and constraints.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from database.connect_db import get_db_connection
from mysql.connector import Error


def extract_machines(cursor):
    """Extract machine information."""
    print("Extracting machine data...")
    
    cursor.execute("""
        SELECT 
            MachineId_i as machine_id,
            MachineName_v as machine_name,
            MachinetypeId_i as machine_type_id,
            1 as is_active  -- Assuming all machines in table are active
        FROM nex_valiant.tbl_machine
        ORDER BY MachineId_i
    """)
    
    machines = cursor.fetchall()
    print(f"  Found {len(machines)} machines")
    
    # Get machine type capabilities (which machine types can do which operations)
    cursor.execute("""
        SELECT DISTINCT MachinetypeId_i
        FROM nex_valiant.tbl_machine
        WHERE MachinetypeId_i IS NOT NULL
        ORDER BY MachinetypeId_i
    """)
    
    machine_types = [row['MachinetypeId_i'] for row in cursor.fetchall()]
    print(f"  Found {len(machine_types)} machine types")
    
    return machines, machine_types


def extract_routing_processes(cursor, limit=100):
    """Extract routing process data (job sequences)."""
    print(f"Extracting routing processes (limit: {limit})...")
    
    try:
        cursor.execute("""
            SELECT 
                BstkId_i as product_id,
                RowId_i as sequence_number,
                ProcessId_i as process_id,
                ProcessDescr_v as process_description,
                MachinetypeId_i as machine_type_id,
                CycleTime_d as cycle_time,
                SetupTime_d as setup_time
            FROM nex_valiant.tbl_routing_process
            WHERE BstkId_i IS NOT NULL
            ORDER BY BstkId_i, RowId_i
            LIMIT %s
        """, (limit,))
        
        routings = cursor.fetchall()
        print(f"  Found {len(routings)} routing processes")
        
        # Group by product
        products = {}
        for route in routings:
            pid = route['product_id']
            if pid not in products:
                products[pid] = []
            products[pid].append(route)
        
        print(f"  Covering {len(products)} unique products")
        return routings
    except Error as e:
        print(f"  Note: Routing extraction failed: {e}")
        return []


def extract_working_hours(cursor):
    """Extract working hours configuration."""
    print("Extracting working hours...")
    
    try:
        cursor.execute("""
            SELECT 
                arrange_day as day_of_week,
                start_time as start_hour,
                end_time as end_hour,
                is_working as is_working_day
            FROM nex_valiant.ai_arrangable_hour
            ORDER BY arrange_day
        """)
        
        working_hours = cursor.fetchall()
        print(f"  Found {len(working_hours)} working hour entries")
        return working_hours
    except Error as e:
        print(f"  Note: Working hours extraction failed: {e}")
        return []


def extract_holidays(cursor):
    """Extract holiday information."""
    print("Extracting holidays...")
    
    try:
        cursor.execute("""
            SELECT 
                holiday_date,
                name as holiday_name,
                is_full_day
            FROM nex_valiant.ai_holidays
            WHERE holiday_date >= CURDATE()
            ORDER BY holiday_date
        """)
        
        holidays = cursor.fetchall()
        print(f"  Found {len(holidays)} upcoming holidays")
        return holidays
    except Error as e:
        print(f"  Note: Holiday extraction failed: {e}")
        return []


def create_snapshot():
    """Create a complete snapshot of production data."""
    conn = get_db_connection()
    
    if conn is None:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Extract all data
        machines, machine_types = extract_machines(cursor)
        routing_processes = extract_routing_processes(cursor, limit=500)
        working_hours = extract_working_hours(cursor)
        holidays = extract_holidays(cursor)
        
        # Create snapshot
        snapshot = {
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                "database": "nex_valiant",
                "version": "1.0",
                "machine_count": len(machines),
                "machine_type_count": len(machine_types),
                "routing_process_count": len(routing_processes)
            },
            "machines": machines,
            "machine_types": machine_types,
            "routing_processes": routing_processes,
            "working_hours": working_hours,
            "holidays": holidays,
            "constraints": {
                "max_working_hours_per_day": 24,
                "setup_time_between_jobs": 0.5,  # 30 minutes
                "maintenance_window": {
                    "day": 7,  # Sunday
                    "start_hour": 6,
                    "duration_hours": 4
                }
            }
        }
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/production_snapshot_{timestamp}.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Write JSON file
        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        print(f"\nSnapshot saved to: {filename}")
        
        # Also save a "latest" version for easy access
        latest_filename = "data/production_snapshot_latest.json"
        with open(latest_filename, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        print(f"Latest snapshot saved to: {latest_filename}")
        
        # Print summary
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"Machines: {len(machines)}")
        print(f"Machine Types: {len(machine_types)}")
        print(f"Routing Processes: {len(routing_processes)}")
        print(f"Working Hours Rules: {len(working_hours)}")
        print(f"Holidays: {len(holidays)}")
        
        cursor.close()
        
    except Error as e:
        print(f"Error during extraction: {e}")
    
    finally:
        if conn.is_connected():
            conn.close()
            print("\nDatabase connection closed.")


def main():
    """Main function."""
    print("Production Data Snapshot Extraction")
    print("="*50)
    print("This will create a static snapshot for training.\n")
    
    create_snapshot()
    
    print("\nNext steps:")
    print("1. Review the generated snapshot file")
    print("2. Use it for training your PPO model")
    print("3. Keep the snapshot for reproducible experiments")


if __name__ == "__main__":
    main()