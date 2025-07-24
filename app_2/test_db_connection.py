#!/usr/bin/env python3
"""
Test database connection and data fetching.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.db_connector import DBConnector
import pprint

def test_database_connection():
    """Test the database connection and fetch sample data."""
    print("Testing Database Connection...")
    print("-" * 50)
    
    # Create connector
    db = DBConnector()
    
    # Test connection
    if db.test_connection():
        print("✓ Database connection successful!")
    else:
        print("✗ Database connection failed!")
        return
    
    try:
        # Connect and get summary
        db.connect()
        
        # Get job count summary
        print("\nJob Count Summary:")
        summary = db.get_job_count_summary()
        pprint.pprint(summary)
        
        # Fetch sample jobs
        print("\nFetching pending jobs...")
        jobs = db.fetch_pending_jobs()
        print(f"Found {len(jobs)} pending jobs")
        
        if jobs:
            print("\nSample job data (first 3):")
            for i, job in enumerate(jobs[:3]):
                print(f"\nJob {i+1}:")
                print(f"  Job ID: {job['job_id']}")
                print(f"  Family ID: {job['family_id']}")
                print(f"  Sequence: {job['sequence']}")
                print(f"  Machine Types: {job['machine_types']}")
                print(f"  Processing Time: {job['processing_time']} hours")
                print(f"  Days to deadline: {job['lcd_days_remaining']}")
                print(f"  Important: {job['is_important']}")
                print(f"  Product: {job['product_code']}")
        
        # Fetch machines
        print("\nFetching machines...")
        machines = db.fetch_machines()
        print(f"Found {len(machines)} machines")
        
        if machines:
            print("\nSample machine data (first 5):")
            for i, machine in enumerate(machines[:5]):
                print(f"  {i+1}. {machine['machine_name']} (Type: {machine['machine_type_id']})")
        
        # Get machine type mapping
        print("\nMachine type mapping:")
        type_map = db.get_machine_type_mapping()
        print(f"Found {len(type_map)} machine name to type mappings")
        
        # Show some mappings
        if type_map:
            print("\nSample mappings:")
            for name, type_id in list(type_map.items())[:5]:
                print(f"  {name} -> Type {type_id}")
        
        # Test working hours
        print("\nFetching working hours...")
        working_hours = db.fetch_working_hours()
        print("Working hours configuration:")
        for day, hours in working_hours.items():
            if day != 'breaks':
                print(f"  {day}: {hours}")
        if 'breaks' in working_hours:
            print(f"  Breaks: {working_hours['breaks']}")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.disconnect()
        print("\nDatabase connection closed.")

if __name__ == "__main__":
    test_database_connection()