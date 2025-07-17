"""
Test database connection and query machine information.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connect_db import get_db_connection
from mysql.connector import Error


def query_machines():
    """Query machine information from the database."""
    conn = get_db_connection()
    
    if conn is None:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Your query
        query = """
        SELECT MachineId_i, MachineName_v, MachinetypeId_i 
        FROM nex_valiant.tbl_machine
        """
        
        cursor.execute(query)
        machines = cursor.fetchall()
        
        print(f"\nTotal machines found: {len(machines)}")
        print("\nFirst 10 machines:")
        print("-" * 80)
        print(f"{'MachineId':<12} {'MachineName':<40} {'MachineTypeId':<12}")
        print("-" * 80)
        
        for i, machine in enumerate(machines[:10]):
            print(f"{machine['MachineId_i']:<12} {machine['MachineName_v']:<40} {machine['MachinetypeId_i']:<12}")
        
        # Show machine type distribution
        print("\n\nMachine Type Distribution:")
        print("-" * 40)
        
        cursor.execute("""
            SELECT MachinetypeId_i, COUNT(*) as count 
            FROM nex_valiant.tbl_machine 
            GROUP BY MachinetypeId_i 
            ORDER BY count DESC
        """)
        
        type_distribution = cursor.fetchall()
        for row in type_distribution:
            print(f"Machine Type {row['MachinetypeId_i']}: {row['count']} machines")
        
        cursor.close()
        
    except Error as e:
        print(f"Error executing query: {e}")
    
    finally:
        if conn.is_connected():
            conn.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    print("Testing database connection and machine query...")
    query_machines()