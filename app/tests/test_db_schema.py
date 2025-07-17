"""
Explore database schema to understand table structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connect_db import get_db_connection
from mysql.connector import Error


def explore_schema():
    """Explore database schema."""
    conn = get_db_connection()
    
    if conn is None:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # List all tables
        print("Tables in nex_valiant database:")
        print("-" * 50)
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = list(table.values())[0]
            print(f"  {table_name}")
        
        # Check specific tables mentioned in CLAUDE.md
        print("\n\nChecking tables mentioned in documentation:")
        print("-" * 50)
        
        tables_to_check = [
            'tbl_machine',
            'tbl_routing_process',
            'ai_arrangable_hour',
            'ai_breaktimes', 
            'ai_holidays',
            'ai_job_sequences'
        ]
        
        for table_name in tables_to_check:
            print(f"\n{table_name}:")
            try:
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                for col in columns[:5]:  # Show first 5 columns
                    print(f"  - {col['Field']} ({col['Type']})")
                if len(columns) > 5:
                    print(f"  ... and {len(columns)-5} more columns")
            except Error as e:
                print(f"  ERROR: {e}")
        
        # Look for job-related tables
        print("\n\nSearching for job-related tables:")
        print("-" * 50)
        cursor.execute("SHOW TABLES LIKE '%job%'")
        job_tables = cursor.fetchall()
        for table in job_tables:
            print(f"  {list(table.values())[0]}")
        
        cursor.execute("SHOW TABLES LIKE '%production%'")
        prod_tables = cursor.fetchall()
        for table in prod_tables:
            print(f"  {list(table.values())[0]}")
            
        cursor.close()
        
    except Error as e:
        print(f"Error exploring schema: {e}")
    
    finally:
        if conn.is_connected():
            conn.close()


if __name__ == "__main__":
    explore_schema()