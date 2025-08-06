"""
Simplified Production Data Ingestion Module for PPO Training
Uses machine names instead of IDs, removes capable_machines complexity
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedDataIngester:
    """Simplified ingester that uses machine names and removes capable_machines."""
    
    def __init__(self,
                 host: str = None,
                 user: str = None,
                 password: str = None,
                 database: str = None,
                 port: int = 3306):
        """Initialize database connection parameters."""
        self.host = host or os.getenv('MARIADB_HOST', 'localhost')
        self.user = user or os.getenv('MARIADB_USERNAME', 'myuser')
        self.password = password or os.getenv('MARIADB_PASSWORD', 'mypassword')
        self.database = database or os.getenv('MARIADB_DATABASE', 'nex_valiant')
        self.port = port or int(os.getenv('MARIADB_PORT', '3306'))
        
    def connect(self) -> pymysql.Connection:
        """Create database connection."""
        try:
            connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                cursorclass=DictCursor,
                charset='utf8mb4'
            )
            logger.info(f"Connected to database: {self.database}@{self.host}")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def fetch_machines(self) -> List[Dict]:
        """Fetch machine names from database."""
        machines = []
        
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                query = """
                SELECT 
                    MachineId_i as machine_id,
                    MachineName_v as machine_name,
                    MachinetypeId_i as machine_type,
                    Status_i as status
                FROM tbl_machine
                WHERE Status_i = 1  -- Active machines only
                ORDER BY MachineName_v
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                for row in results:
                    machines.append({
                        'machine_id': row['machine_id'],
                        'machine_name': row['machine_name'],
                        'machine_type': row['machine_type']
                    })
                
                logger.info(f"Fetched {len(machines)} active machines")
                return machines
                
        except Exception as e:
            logger.error(f"Error fetching machines: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
                
    def fetch_job_data(self,
                      planning_horizon_days: int = 30,
                      job_family_limit: Optional[int] = None) -> List[Dict]:
        """Fetch job data from database."""
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                query = """
                SELECT
                    jot.TxnId_i,
                    jot.DocRef_v AS job,
                    jot.TargetDate_dd AS lcd_date,
                    jop.Task_v AS process_code,
                    jop.RowId_i AS process_sequence,
                    jop.QtyStatus_c AS process_status,
                    jot.JoQty_d AS job_quantity,
                    
                    -- Machine assignment
                    jop.Machine_v AS assigned_machine_id,
                    tm.MachineName_v AS assigned_machine_name,
                    tm.MachinetypeId_i AS machine_type,
                    
                    -- Time calculations
                    CASE WHEN jop.CapMin_d = 1 AND jop.CapQty_d != 0 
                         THEN jot.JoQty_d / (jop.CapQty_d * 60)
                         WHEN jop.LeadTime_d IS NOT NULL AND jop.LeadTime_d > 0
                         THEN jop.LeadTime_d * 8
                         ELSE 2.0 END AS hours_need,
                    
                    jop.SetupTime_d AS setting_hours,
                    
                    -- Progress tracking
                    SUM(di.Qty_d) AS accumulated_output,
                    (jot.JoQty_d - COALESCE(SUM(di.Qty_d), 0)) AS balance_quantity,
                    
                    jot.MaterialDate_dd AS material_arrival,
                    jot.CreateDate_dt AS plan_date

                FROM tbl_jo_process AS jop
                    INNER JOIN tbl_jo_txn AS jot
                        ON jot.TxnId_i = jop.TxnId_i 
                    
                    LEFT JOIN tbl_daily_item AS di
                        ON di.JoId_i = jop.TxnId_i 
                        AND di.ProcessrowId_i = jop.RowId_i 
                        AND di.CreateDate_dt >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                    
                    LEFT JOIN tbl_machine AS tm
                        ON tm.MachineId_i = CAST(jop.Machine_v AS UNSIGNED)

                WHERE jot.Void_c != 1
                    AND jot.DocStatus_c NOT IN ('CP', 'CX')
                    AND jop.QtyStatus_c != 'FF'
                    AND jot.TargetDate_dd > CURDATE()
                    AND jot.TargetDate_dd <= DATE_ADD(CURDATE(), INTERVAL %s DAY)
                    AND jot.CreateDate_dt >= DATE_SUB(CURDATE(), INTERVAL 100 DAY)
                    AND jot.MaterialDate_dd IS NOT NULL
                    AND jot.MaterialDate_dd <= CURDATE()

                GROUP BY
                    jop.TxnId_i, jop.RowId_i

                ORDER BY 
                    jot.TargetDate_dd ASC,
                    jop.TxnId_i ASC,
                    jop.RowId_i ASC

                LIMIT %s
                """
                
                limit = job_family_limit * 10 if job_family_limit else 1000
                cursor.execute(query, (planning_horizon_days, limit))
                results = cursor.fetchall()
                
                logger.info(f"Fetched {len(results)} job processes")
                return results
                
        except Exception as e:
            logger.error(f"Error fetching job data: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
                
    def convert_to_simple_format(self,
                                 job_data: List[Dict],
                                 machines: List[Dict],
                                 job_family_limit: Optional[int] = None) -> Dict[str, Dict]:
        """Convert to simplified format without capable_machines."""
        families = {}
        
        # Create machine name lookup
        machine_lookup = {m['machine_id']: m['machine_name'] for m in machines}
        
        # Group by job reference (family)
        job_groups = {}
        for row in job_data:
            family_id = row['job'] or f"TXN_{row['TxnId_i']}"
            
            if family_id not in job_groups:
                job_groups[family_id] = []
            job_groups[family_id].append(row)
        
        # Apply job family limit if specified
        if job_family_limit and job_family_limit < len(job_groups):
            # Sort by LCD date (most urgent first)
            sorted_families = sorted(job_groups.items(), 
                                   key=lambda x: x[1][0].get('lcd_date', datetime.max.date()))
            job_groups = dict(sorted_families[:job_family_limit])
            logger.info(f"Limited to {job_family_limit} most urgent job families")
        
        # Process each family
        for family_id, processes in job_groups.items():
            # Sort by process sequence
            processes.sort(key=lambda x: x.get('process_sequence', 0))
            
            # Get family-level info from first process
            first_process = processes[0]
            
            # Calculate LCD days remaining
            lcd_date = first_process['lcd_date']
            if lcd_date:
                days_remaining = (lcd_date - datetime.now().date()).days
            else:
                days_remaining = 30
            
            # Build task list
            tasks = []
            for idx, process in enumerate(processes):
                # Calculate processing time
                processing_time = float(process['hours_need'] or 2.0)
                if process['setting_hours']:
                    processing_time += float(process['setting_hours'])
                
                # Get assigned machine name
                assigned_machine = None
                if process['assigned_machine_id'] and process['assigned_machine_id'] in machine_lookup:
                    assigned_machine = machine_lookup[process['assigned_machine_id']]
                elif process['assigned_machine_name']:
                    assigned_machine = process['assigned_machine_name']
                
                task = {
                    'sequence': idx + 1,
                    'process_name': process['process_code'],
                    'processing_time': round(processing_time, 2),
                    'assigned_machine': assigned_machine,  # Machine name or None
                    'balance_quantity': float(process['balance_quantity'] or 0),
                    'original_quantity': float(process['job_quantity'] or 0)
                }
                
                tasks.append(task)
            
            # Create family entry
            families[family_id] = {
                'job_reference': family_id,
                'lcd_date': lcd_date.isoformat() if lcd_date else None,
                'lcd_days_remaining': max(0, days_remaining),
                'is_urgent': days_remaining <= 7,
                'total_tasks': len(tasks),
                'tasks': tasks,
                'material_arrival': first_process['material_arrival'].isoformat() if first_process['material_arrival'] else None,
                'created_date': first_process['plan_date'].isoformat() if first_process['plan_date'] else None
            }
        
        logger.info(f"Converted {len(families)} job families with {sum(len(f['tasks']) for f in families.values())} total tasks")
        return families
        
    def create_snapshot(self,
                       planning_horizon_days: int = 30,
                       output_file: str = None,
                       job_family_limit: Optional[int] = None) -> str:
        """Create a simplified snapshot for PPO training."""
        if output_file is None:
            if job_family_limit:
                output_file = f'/Users/carrickcheah/Project/ppo/app3/data/{job_family_limit}_jobs.json'
            else:
                output_file = f'/Users/carrickcheah/Project/ppo/app3/data/all_jobs.json'
        
        # Fetch data
        logger.info("Fetching machine data...")
        machines = self.fetch_machines()
        
        logger.info("Fetching job data...")
        job_data = self.fetch_job_data(
            planning_horizon_days=planning_horizon_days,
            job_family_limit=job_family_limit
        )
        
        logger.info("Converting to simplified format...")
        families = self.convert_to_simple_format(job_data, machines, job_family_limit)
        
        # Create snapshot
        snapshot = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'database': self.database,
                'planning_horizon_days': planning_horizon_days,
                'total_families': len(families),
                'total_tasks': sum(len(f['tasks']) for f in families.values()),
                'total_machines': len(machines)
            },
            'families': families,
            'machines': [m['machine_name'] for m in machines]  # Just machine names
        }
        
        # Add statistics
        snapshot['statistics'] = {
            'urgent_families': sum(1 for f in families.values() if f['is_urgent']),
            'critical_families': sum(1 for f in families.values() if f['lcd_days_remaining'] <= 3),
            'total_tasks': sum(len(f['tasks']) for f in families.values())
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        logger.info(f"Saved simplified snapshot to {output_file}")
        return output_file


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create simplified snapshots with machine names")
    parser.add_argument('--job-limit', type=int, help='Number of job families')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    ingester = SimplifiedDataIngester()
    
    output_file = ingester.create_snapshot(
        job_family_limit=args.job_limit,
        output_file=args.output
    )
    
    print(f"✓ Created snapshot: {output_file}")