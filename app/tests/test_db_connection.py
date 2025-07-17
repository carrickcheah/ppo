#!/usr/bin/env python3
import sys
sys.path.append('.')

from database.connect_db import get_db_connection

def test_connection():
    """Test the database connection"""
    print("Testing MariaDB connection...")
    
    conn = get_db_connection()
    if conn:
        print("✓ Successfully connected to MariaDB!")
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"✓ MariaDB version: {version[0]}")
        
        # List available tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"\n✓ Found {len(tables)} tables in database:")
        for table in tables[:10]:  # Show first 10 tables
            print(f"  - {table[0]}")
        if len(tables) > 10:
            print(f"  ... and {len(tables) - 10} more")
        
        cursor.close()
        conn.close()
        print("\n✓ Connection closed successfully")
    else:
        print("✗ Failed to connect to MariaDB")
        print("\nPlease check your .env file has correct credentials:")
        print("  MARIADB_HOST, MARIADB_USERNAME, MARIADB_PASSWORD, MARIADB_DATABASE, MARIADB_PORT")

if __name__ == "__main__":
    test_connection()