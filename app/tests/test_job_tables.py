"""
Explore job-related tables to understand production data structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connect_db import get_db_connection
from mysql.connector import Error


def explore_job_tables():
    """Explore job order tables."""
    conn = get_db_connection()
    
    if conn is None:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Check tbl_jo_txn structure
        print("Job Order Transaction Table (tbl_jo_txn):")
        print("-" * 80)
        cursor.execute("DESCRIBE tbl_jo_txn")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col['Field']:<30} {col['Type']:<20} {col['Null']:<5} {col['Key']}")
        
        # Get sample job orders
        print("\n\nSample Job Orders (first 5):")
        print("-" * 80)
        cursor.execute("""
            SELECT * FROM tbl_jo_txn 
            ORDER BY JoId_i DESC 
            LIMIT 5
        """)
        jobs = cursor.fetchall()
        
        if jobs:
            # Print column headers
            cols = list(jobs[0].keys())
            for col in cols[:6]:  # First 6 columns
                print(f"{col:<20}", end=" ")
            print("\n" + "-" * 130)
            
            # Print data
            for job in jobs:
                for col in cols[:6]:
                    val = str(job[col])[:19]
                    print(f"{val:<20}", end=" ")
                print()
        
        # Check tbl_jo_item structure
        print("\n\nJob Order Items Table (tbl_jo_item):")
        print("-" * 80)
        cursor.execute("DESCRIBE tbl_jo_item")
        columns = cursor.fetchall()
        for col in columns[:10]:  # First 10 columns
            print(f"  {col['Field']:<30} {col['Type']:<20}")
        
        # Check routing process
        print("\n\nSample Routing Process:")
        print("-" * 80)
        cursor.execute("""
            SELECT Version_v, BstkId_i, ProcessId_i, RowId_i, ProcessDescr_v
            FROM tbl_routing_process
            LIMIT 5
        """)
        routings = cursor.fetchall()
        for routing in routings:
            print(f"  Version: {routing['Version_v']}, "
                  f"Process: {routing['ProcessDescr_v'][:50]}, "
                  f"Row: {routing['RowId_i']}")
        
        # Check view
        print("\n\nView Job Data (vw_job_data):")
        print("-" * 80)
        cursor.execute("DESCRIBE vw_job_data")
        columns = cursor.fetchall()
        for col in columns[:10]:
            print(f"  {col['Field']:<30} {col['Type']:<20}")
            
        cursor.close()
        
    except Error as e:
        print(f"Error exploring job tables: {e}")
    
    finally:
        if conn.is_connected():
            conn.close()


if __name__ == "__main__":
    explore_job_tables()