"""
Get actual production data structure for PPO training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connect_db import get_db_connection
from mysql.connector import Error


def get_production_data():
    """Get production data for training."""
    conn = get_db_connection()
    
    if conn is None:
        print("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get recent job orders
        print("Recent Job Orders:")
        print("-" * 100)
        cursor.execute("""
            SELECT 
                TxnId_i,
                ItemId_i,
                BstkId_i,
                DocRef_v,
                TxnDate_dd,
                TargetDate_dd,
                JoQty_d,
                QtyDone_d,
                DocStatus_c
            FROM tbl_jo_txn
            WHERE Void_c = 'N'
            AND DocStatus_c IN ('CF', 'NC')  -- Confirmed or Not Confirmed
            ORDER BY TxnDate_dd DESC
            LIMIT 10
        """)
        
        jobs = cursor.fetchall()
        for job in jobs:
            print(f"TxnId: {job['TxnId_i']}, "
                  f"Item: {job['ItemId_i']}, "
                  f"Ref: {job['DocRef_v']}, "
                  f"Target: {job['TargetDate_dd']}, "
                  f"Qty: {job['JoQty_d']}/{job['QtyDone_d']}, "
                  f"Status: {job['DocStatus_c']}")
        
        # Get job items (details)
        print("\n\nJob Order Items (first job details):")
        print("-" * 100)
        
        if jobs:
            first_job_id = jobs[0]['TxnId_i']
            cursor.execute("""
                SELECT 
                    JoId_i,
                    RowId_i,
                    ProcessId_i,
                    MachineId_i,
                    LeadTime_i,
                    QtyDone_d,
                    _Status_c
                FROM tbl_jo_item
                WHERE JoId_i = %s
                ORDER BY RowId_i
            """, (first_job_id,))
            
            items = cursor.fetchall()
            for item in items:
                print(f"  Row: {item['RowId_i']}, "
                      f"Process: {item['ProcessId_i']}, "
                      f"Machine: {item['MachineId_i']}, "
                      f"Lead: {item['LeadTime_i']}h, "
                      f"Done: {item['QtyDone_d']}, "
                      f"Status: {item['_Status_c']}")
        
        # Get routing information
        print("\n\nRouting Process Examples:")
        print("-" * 100)
        cursor.execute("""
            SELECT 
                BstkId_i,
                ProcessId_i,
                RowId_i,
                ProcessDescr_v
            FROM tbl_routing_process
            WHERE RowId_i <= 5
            LIMIT 10
        """)
        
        routings = cursor.fetchall()
        for route in routings:
            print(f"Product: {route['BstkId_i']}, "
                  f"Step: {route['RowId_i']}, "
                  f"Process: {route['ProcessId_i']}, "
                  f"Desc: {route['ProcessDescr_v'][:50] if route['ProcessDescr_v'] else 'N/A'}")
        
        # Count active jobs
        cursor.execute("""
            SELECT 
                COUNT(*) as total_jobs,
                SUM(CASE WHEN DocStatus_c = 'CF' THEN 1 ELSE 0 END) as confirmed,
                SUM(CASE WHEN QtyDone_d < JoQty_d THEN 1 ELSE 0 END) as in_progress
            FROM tbl_jo_txn
            WHERE Void_c = 'N'
            AND DocStatus_c IN ('CF', 'NC')
        """)
        
        stats = cursor.fetchone()
        print(f"\n\nJob Statistics:")
        print(f"Total Active Jobs: {stats['total_jobs']}")
        print(f"Confirmed: {stats['confirmed']}")
        print(f"In Progress: {stats['in_progress']}")
        
        cursor.close()
        
    except Error as e:
        print(f"Error getting production data: {e}")
    
    finally:
        if conn.is_connected():
            conn.close()


if __name__ == "__main__":
    get_production_data()