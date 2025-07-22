#!/usr/bin/env python3
"""
Test production database connection before deployment
"""

import os
import sys
from dotenv import load_dotenv
import pymysql
from datetime import datetime

# Load environment variables
load_dotenv()

print("Production Connection Test")
print("=" * 50)

# Test database connection
print("\n1. Testing Database Connection...")
try:
    # Try both sets of credentials
    configs = [
        {
            "name": "DB_* variables",
            "host": os.getenv('DB_HOST', 'localhost'),
            "user": os.getenv('DB_USER', 'myuser'),
            "password": os.getenv('DB_PASSWORD', 'mypassword'),
            "database": os.getenv('DB_NAME', 'nex_valiant'),
            "port": int(os.getenv('DB_PORT', '3306'))
        },
        {
            "name": "MARIADB_* variables",
            "host": os.getenv('MARIADB_HOST', 'localhost'),
            "user": os.getenv('MARIADB_USERNAME', 'myuser'),
            "password": os.getenv('MARIADB_PASSWORD', 'mypassword'),
            "database": os.getenv('MARIADB_DATABASE', 'nex_valiant'),
            "port": int(os.getenv('MARIADB_PORT', '3306'))
        }
    ]
    
    for config in configs:
        print(f"\nTrying {config['name']}...")
        print(f"  Host: {config['host']}")
        print(f"  User: {config['user']}")
        print(f"  Database: {config['database']}")
        print(f"  Port: {config['port']}")
        
        try:
            connection = pymysql.connect(
                host=config['host'],
                user=config['user'],
                password=config['password'],
                database=config['database'],
                port=config['port'],
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with connection.cursor() as cursor:
                # Test machine table
                cursor.execute("SELECT COUNT(*) as count FROM tbl_machine WHERE IsActive_b = 1")
                machine_count = cursor.fetchone()['count']
                print(f"  ✓ Connected! Found {machine_count} active machines")
                
                # Test job tables
                cursor.execute("""
                    SELECT COUNT(DISTINCT WorkOrderId_v) as job_count 
                    FROM job_operations_time 
                    WHERE Status_i IN (1,2,6)
                """)
                job_count = cursor.fetchone()['count']
                print(f"  ✓ Found {job_count} pending jobs")
                
            connection.close()
            print(f"  ✓ Database connection successful!")
            
        except Exception as e:
            print(f"  ✗ Connection failed: {str(e)}")
            
except Exception as e:
    print(f"Error: {e}")

# Test API configuration
print("\n2. Checking API Configuration...")
print(f"  API Host: {os.getenv('API_HOST', '0.0.0.0')}")
print(f"  API Port: {os.getenv('API_PORT', '8000')}")
print(f"  API Key: {'*' * 10 if os.getenv('API_KEY') else 'NOT SET'}")
print(f"  Environment: {os.getenv('ENVIRONMENT', 'development')}")
print(f"  CORS Origins: {os.getenv('CORS_ALLOW_ORIGINS', '[]')}")

# Check if using production values
print("\n3. Security Check...")
warnings = []

if os.getenv('API_KEY') == 'dev-api-key-change-in-production':
    warnings.append("⚠️  Using default development API key - CHANGE FOR PRODUCTION")
    
if os.getenv('DB_PASSWORD') == 'mypassword':
    warnings.append("⚠️  Using default database password - CHANGE FOR PRODUCTION")
    
if os.getenv('ENVIRONMENT') == 'development':
    warnings.append("⚠️  Environment set to 'development' - Change to 'production'")
    
if 'localhost' in os.getenv('CORS_ALLOW_ORIGINS', ''):
    warnings.append("⚠️  CORS allows localhost - Update for production domains")

if warnings:
    print("\nSecurity Warnings:")
    for warning in warnings:
        print(f"  {warning}")
else:
    print("  ✓ No security warnings found")

print("\n" + "=" * 50)
print("Test complete!")

if warnings:
    print("\n⚠️  IMPORTANT: Address security warnings before production deployment!")
else:
    print("\n✓ System appears ready for production deployment!")
    print("\nNext steps:")
    print("1. Start the API server: uv run python run_api_server.py")
    print("2. Access API docs: http://localhost:8000/docs")
    print("3. Begin shadow mode testing with real data")