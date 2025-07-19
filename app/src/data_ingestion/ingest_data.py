"""
Production Data Ingestion Module for PPO Training
Converts MariaDB production data to PPO-compatible format
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


class ProductionDataIngester:
    """Ingests production data from MariaDB and converts to PPO format."""
    
    def __init__(self,
                 host: str = None,
                 user: str = None,
                 password: str = None,
                 database: str = None,
                 port: int = 3306):
        """Initialize database connection parameters."""
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.user = user or os.getenv('DB_USER', 'root')
        self.password = password or os.getenv('DB_PASSWORD', '')
        self.database = database or os.getenv('DB_NAME', 'production')
        self.port = port or int(os.getenv('DB_PORT', '3306'))
        
        # SQL query for fetching job data
        self.job_query = self._build_job_query()
        
    def _build_job_query(self) -> str:
        """Build the SQL query for fetching job data."""
        return """
        SELECT
            -- Job planning and timing information
            jot.CreateDate_dt AS plan_date,
            jot.TargetDate_dd AS lcd_date,
            jop.TxnId_i,
            jot.DocRef_v AS job,
            jop.Task_v AS process_code,
            
            -- Resource and machine information
            '' AS rsc_location,
            tm.MachineId_i,
            tm.MachineName_v,
            tm.MachinetypeId_i,
            jop.ManCount_i AS number_operator,
            
            -- Quantity calculations
            jot.JoQty_d AS job_quantity,
            
            -- Expected output per hour calculation
            CASE WHEN jop.CapMin_d = 1 AND jop.CapQty_d != 0 
                 THEN jop.CapQty_d * 60
                 ELSE NULL END AS expect_output_per_hour,
            
            -- Hours needed calculation with multiple scenarios
            CASE WHEN jop.CapMin_d = 1 AND jop.CapQty_d != 0 
                 THEN jot.JoQty_d / (jop.CapQty_d * 60)
                 WHEN (jop.Machine_v IS NULL OR jop.Machine_v = '' OR jop.Machine_v = '0') 
                      AND jop.LeadTime_d IS NOT NULL AND jop.LeadTime_d > 0
                 THEN jop.LeadTime_d * %s
                 ELSE NULL END AS hours_need,
            
            -- Days needed calculation with fallback logic
            CASE WHEN jop.CapMin_d = 1 AND jop.CapQty_d != 0 
                 THEN jot.JoQty_d / (jop.CapQty_d * 60 * 24)
                 WHEN jop.CapMin_d = 0 AND jop.LeadTime_d IS NOT NULL 
                      AND jop.LeadTime_d > 0
                 THEN jop.LeadTime_d
                 WHEN (jop.Machine_v IS NULL OR jop.Machine_v = '' 
                      OR jop.Machine_v = '0') AND jop.LeadTime_d IS NOT NULL 
                      AND jop.LeadTime_d > 0
                 THEN jop.LeadTime_d
                 ELSE NULL END AS day_need,
            
            -- Additional planning fields
            jop.SetupTime_d AS setting_hours,
            '' AS start_date,
            SUM(di.Qty_d) AS accumulated_daily_output,
            (jot.JoQty_d - COALESCE(SUM(di.Qty_d), 0)) AS balance_quantity,
            jot.MaterialDate_dd AS material_arrival,
            %s AS priority,
            NOW() AS created_at,
            NOW() AS updated_at,
            
            -- Process sequence information
            jop.RowId_i AS process_row_id,
            jop.QtyStatus_c AS process_status,
            jop.RowId_i AS process_sequence  -- Use RowId_i for sequence ordering

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
            jop.TxnId_i, jop.RowId_i, jot.CreateDate_dt, jot.TargetDate_dd, 
            jot.DocRef_v, jop.Task_v, tm.MachineId_i, tm.MachineName_v, 
            tm.MachinetypeId_i, jop.ManCount_i, jot.JoQty_d, jop.CapQty_d, 
            jop.CapMin_d, jop.LeadTime_d, jop.SetupTime_d, jot.MaterialDate_dd, 
            jop.Machine_v, jop.QtyStatus_c

        ORDER BY 
            jot.CreateDate_dt DESC,
            jop.TxnId_i ASC,
            jop.RowId_i ASC

        LIMIT %s
        """
        
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
            
    def fetch_machine_data(self) -> Dict[int, Dict]:
        """Fetch machine information from database."""
        machines = {}
        
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                query = """
                SELECT 
                    MachineId_i as machine_id,
                    MachineName_v as machine_name,
                    MachinetypeId_i as machine_type_id,
                    Status_i as status
                FROM tbl_machine
                WHERE Status_i = 1  -- Active machines (assuming 1 = active)
                ORDER BY MachineId_i
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                for row in results:
                    machines[row['machine_id']] = {
                        'machine_id': row['machine_id'],
                        'machine_name': row['machine_name'],
                        'machine_type_id': row['machine_type_id'],
                        'status': row['status'],
                        'efficiency_rate': 1.0  # Default efficiency rate since column doesn't exist
                    }
                
                logger.info(f"Fetched {len(machines)} active machines")
                return machines
                
        except Exception as e:
            logger.error(f"Error fetching machines: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
                
    def fetch_machine_type_capabilities(self) -> Dict[int, List[str]]:
        """Fetch machine type to process code mapping."""
        capabilities = {}
        
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                # Get machine type to process mapping
                # This is a simplified version - you may need to adjust based on your actual schema
                query = """
                SELECT DISTINCT
                    tm.MachinetypeId_i,
                    jop.Task_v AS process_code
                FROM tbl_machine tm
                INNER JOIN tbl_jo_process jop 
                    ON tm.MachineId_i = CAST(jop.Machine_v AS UNSIGNED)
                WHERE tm.Status_i = 1
                    AND jop.Task_v IS NOT NULL
                    AND jop.Task_v != ''
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Build capability map
                for row in results:
                    machine_type = row['MachinetypeId_i']
                    process_code = row['process_code']
                    
                    if machine_type not in capabilities:
                        capabilities[machine_type] = []
                    
                    # Extract process prefix (e.g., "CH04" from "CH04-002-2/4")
                    process_prefix = process_code.split('-')[0] if '-' in process_code else process_code
                    
                    if process_prefix not in capabilities[machine_type]:
                        capabilities[machine_type].append(process_prefix)
                
                logger.info(f"Built capability map for {len(capabilities)} machine types")
                return capabilities
                
        except Exception as e:
            logger.error(f"Error fetching machine capabilities: {e}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()
                
    def fetch_job_data(self,
                      planning_horizon_days: int = 30,
                      manual_hours_multiplier: float = 8.0,
                      priority_default: int = 5,
                      limit: int = 1000) -> List[Dict]:
        """Fetch job data from database."""
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                cursor.execute(
                    self.job_query,
                    (manual_hours_multiplier, priority_default, planning_horizon_days, limit)
                )
                results = cursor.fetchall()
                logger.info(f"Fetched {len(results)} job processes")
                return results
                
        except Exception as e:
            logger.error(f"Error fetching job data: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
                
    def convert_to_ppo_format(self,
                             job_data: List[Dict],
                             machines: Dict[int, Dict],
                             machine_capabilities: Dict[int, List[str]]) -> Dict[str, Dict]:
        """Convert raw job data to PPO-compatible format."""
        families = {}
        
        # Group by job reference (family)
        job_groups = {}
        for row in job_data:
            # Use DocRef_v (job reference) as family ID
            family_id = row['job']  # This is jot.DocRef_v from the query
            if not family_id:
                # Fallback to transaction ID if job reference is missing
                family_id = f"TXN_{row['TxnId_i']}"
            
            if family_id not in job_groups:
                job_groups[family_id] = []
            job_groups[family_id].append(row)
        
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
                days_remaining = 30  # Default if no LCD date
            
            # Determine if job is important based on priority or days remaining
            is_important = (
                first_process.get('priority', 5) >= 8 or
                days_remaining <= 7 or
                'URGENT' in first_process.get('job', '').upper() or
                'RUSH' in first_process.get('job', '').upper()
            )
            
            # Build task list
            tasks = []
            for idx, process in enumerate(processes):
                # Calculate processing time in hours
                if process['hours_need'] is not None:
                    processing_time = float(process['hours_need'])
                elif process['day_need'] is not None:
                    processing_time = float(process['day_need']) * 24  # Convert days to hours
                else:
                    processing_time = 2.0  # Default 2 hours
                
                # Add setup time if specified
                if process['setting_hours']:
                    processing_time += float(process['setting_hours'])
                
                # Determine capable machines
                capable_machines = []
                
                # If machine is specified, use its type to find similar machines
                if process['MachineId_i']:
                    assigned_machine = process['MachineId_i']
                    if assigned_machine in machines:
                        machine_type = machines[assigned_machine]['machine_type_id']
                        
                        # Find all machines of the same type
                        for m_id, m_info in machines.items():
                            if m_info['machine_type_id'] == machine_type:
                                capable_machines.append(m_id)
                else:
                    # Manual process - use all machines that can handle the process code
                    process_prefix = process['process_code'].split('-')[0] if process['process_code'] else ''
                    
                    for machine_type, capabilities in machine_capabilities.items():
                        if process_prefix in capabilities:
                            # Add all machines of this type
                            for m_id, m_info in machines.items():
                                if m_info['machine_type_id'] == machine_type:
                                    capable_machines.append(m_id)
                    
                    # If no capable machines found, allow all machines (manual process)
                    if not capable_machines:
                        capable_machines = list(machines.keys())
                
                # Determine task status
                if process['balance_quantity'] == 0:
                    status = 'completed'
                elif process['accumulated_daily_output'] and process['accumulated_daily_output'] > 0:
                    status = 'in_progress'
                else:
                    status = 'pending'
                
                task = {
                    'sequence': idx + 1,  # 1-based sequence
                    'process_name': process['process_code'],
                    'processing_time': round(processing_time, 2),
                    'capable_machines': sorted(list(set(capable_machines))),  # Remove duplicates
                    'status': status,
                    'balance_quantity': float(process['balance_quantity'] or 0),
                    'original_quantity': float(process['job_quantity'] or 0)
                }
                
                tasks.append(task)
            
            # Create family entry
            families[family_id] = {
                'transaction_id': first_process['TxnId_i'],  # Keep original TxnId_i for reference
                'job_reference': family_id,  # This is now the DocRef_v
                'product': first_process['process_code'].split('-')[0] if first_process['process_code'] else 'UNKNOWN',
                'is_important': is_important,
                'lcd_date': lcd_date.isoformat() if lcd_date else None,
                'lcd_days_remaining': max(0, days_remaining),
                'total_sequences': len(tasks),
                'tasks': tasks,
                'material_arrival': first_process['material_arrival'].isoformat() if first_process['material_arrival'] else None,
                'created_date': first_process['plan_date'].isoformat() if first_process['plan_date'] else None
            }
        
        logger.info(f"Converted {len(families)} job families with {sum(len(f['tasks']) for f in families.values())} total tasks")
        return families
        
    def create_snapshot(self,
                       planning_horizon_days: int = 30,
                       output_file: str = None,
                       include_machines: bool = True) -> str:
        """Create a complete snapshot for PPO training."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'app/data/production_snapshot_{timestamp}.json'
        
        # Fetch all required data
        logger.info("Fetching machine data...")
        machines = self.fetch_machine_data()
        
        logger.info("Fetching machine capabilities...")
        machine_capabilities = self.fetch_machine_type_capabilities()
        
        logger.info("Fetching job data...")
        job_data = self.fetch_job_data(planning_horizon_days=planning_horizon_days)
        
        logger.info("Converting to PPO format...")
        families = self.convert_to_ppo_format(job_data, machines, machine_capabilities)
        
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
            'families': families
        }
        
        if include_machines:
            # Convert machine dict to list format for compatibility
            machine_list = []
            for m_id, m_info in machines.items():
                machine_list.append({
                    'machine_id': m_id,
                    'machine_name': m_info['machine_name'],
                    'machine_type_id': m_info['machine_type_id'],
                    'is_active': 1 if m_info['status'] == 1 else 0
                })
            snapshot['machines'] = machine_list
        
        # Add statistics
        snapshot['statistics'] = {
            'urgent_families': sum(1 for f in families.values() if f['lcd_days_remaining'] <= 7),
            'critical_families': sum(1 for f in families.values() if f['lcd_days_remaining'] <= 3),
            'pending_tasks': sum(1 for f in families.values() for t in f['tasks'] if t['status'] == 'pending'),
            'in_progress_tasks': sum(1 for f in families.values() for t in f['tasks'] if t['status'] == 'in_progress'),
            'completed_tasks': sum(1 for f in families.values() for t in f['tasks'] if t['status'] == 'completed')
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        logger.info(f"Saved snapshot to {output_file}")
        return output_file
        
    def get_production_metrics(self) -> Dict[str, Any]:
        """Get current production metrics for monitoring."""
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                metrics = {}
                
                # Total active jobs
                cursor.execute("""
                    SELECT COUNT(DISTINCT TxnId_i) as active_jobs
                    FROM tbl_jo_txn
                    WHERE Void_c != 1
                        AND DocStatus_c NOT IN ('CP', 'CX')
                        AND TargetDate_dd > CURDATE()
                """)
                metrics['active_jobs'] = cursor.fetchone()['active_jobs']
                
                # Jobs by urgency
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN DATEDIFF(TargetDate_dd, CURDATE()) <= 3 THEN 'critical'
                            WHEN DATEDIFF(TargetDate_dd, CURDATE()) <= 7 THEN 'urgent'
                            WHEN DATEDIFF(TargetDate_dd, CURDATE()) <= 14 THEN 'medium'
                            ELSE 'normal'
                        END as urgency,
                        COUNT(*) as count
                    FROM tbl_jo_txn
                    WHERE Void_c != 1
                        AND DocStatus_c NOT IN ('CP', 'CX')
                        AND TargetDate_dd > CURDATE()
                    GROUP BY urgency
                """)
                
                urgency_counts = {row['urgency']: row['count'] for row in cursor.fetchall()}
                metrics['jobs_by_urgency'] = urgency_counts
                
                # Machine utilization
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_machines,
                        SUM(CASE WHEN Status_c = 'A' THEN 1 ELSE 0 END) as active_machines
                    FROM tbl_machine
                """)
                machine_stats = cursor.fetchone()
                metrics['machine_stats'] = {
                    'total': machine_stats['total_machines'],
                    'active': machine_stats['active_machines']
                }
                
                # Process completion rate (last 7 days)
                cursor.execute("""
                    SELECT 
                        COUNT(CASE WHEN QtyStatus_c = 'FF' THEN 1 END) as completed,
                        COUNT(*) as total
                    FROM tbl_jo_process
                    WHERE CreateDate_dt >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
                """)
                completion_stats = cursor.fetchone()
                metrics['completion_rate'] = {
                    'completed': completion_stats['completed'],
                    'total': completion_stats['total'],
                    'rate': completion_stats['completed'] / completion_stats['total'] if completion_stats['total'] > 0 else 0
                }
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest production data for PPO training")
    parser.add_argument('--host', help='Database host')
    parser.add_argument('--user', help='Database user')
    parser.add_argument('--password', help='Database password')
    parser.add_argument('--database', help='Database name')
    parser.add_argument('--port', type=int, default=3306, help='Database port')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--horizon', type=int, default=30, help='Planning horizon in days')
    parser.add_argument('--metrics', action='store_true', help='Show production metrics')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    
    args = parser.parse_args()
    
    # Create ingester
    ingester = ProductionDataIngester(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
        port=args.port
    )
    
    if args.test:
        # Test connection
        try:
            conn = ingester.connect()
            print("✓ Database connection successful")
            conn.close()
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            exit(1)
            
    elif args.metrics:
        # Show metrics
        try:
            metrics = ingester.get_production_metrics()
            print("\n=== Production Metrics ===")
            print(f"Active Jobs: {metrics['active_jobs']}")
            print("\nJobs by Urgency:")
            for urgency, count in metrics.get('jobs_by_urgency', {}).items():
                print(f"  {urgency.capitalize()}: {count}")
            print(f"\nMachines: {metrics['machine_stats']['active']}/{metrics['machine_stats']['total']} active")
            print(f"\nCompletion Rate (7 days): {metrics['completion_rate']['completed']}/{metrics['completion_rate']['total']} ({metrics['completion_rate']['rate']:.1%})")
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            exit(1)
    else:
        # Create snapshot
        try:
            output_file = ingester.create_snapshot(
                planning_horizon_days=args.horizon,
                output_file=args.output
            )
            print(f"✓ Snapshot created: {output_file}")
        except Exception as e:
            print(f"✗ Error creating snapshot: {e}")
            exit(1)