"""
MariaDB Database Connector

Simple connector to fetch pending jobs and machine data from production database.
No validation or transformation - just raw data extraction.
"""

import pymysql
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DBConnector:
    """
    Simple MariaDB connector for fetching production data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize database connector.
        
        Args:
            config: Database configuration (if None, uses environment variables)
        """
        if config:
            self.host = config.get('host', 'localhost')
            self.port = config.get('port', 3306)
            self.user = config.get('user', 'myuser')
            self.password = config.get('password', 'mypassword')
            self.database = config.get('database', 'nex_valiant')
        else:
            # Use environment variables
            self.host = os.getenv('DB_HOST', 'localhost')
            self.port = int(os.getenv('DB_PORT', '3306'))
            self.user = os.getenv('DB_USER', 'myuser')
            self.password = os.getenv('DB_PASSWORD', 'mypassword')
            self.database = os.getenv('DB_NAME', 'nex_valiant')
            
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"Connected to MariaDB at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from MariaDB")
            
    def fetch_pending_jobs(self) -> List[Dict[str, Any]]:
        """
        Fetch all pending jobs from database.
        
        Jobs are grouped by TxnId_i (transaction), with multiple processes (RowId_i) per transaction.
        We treat each TxnId as a job family, with RowId as the sequence.
        
        Returns:
            List of job dictionaries with fields:
            - job_id: Unique job identifier (DocRef_Task)
            - family_id: Job family identifier (DocRef)
            - sequence: Process sequence within job (RowId)
            - required_machines: List of machine IDs that must ALL be used simultaneously
            - processing_time: Processing time in hours (calculated from capacity)
            - lcd_date: Target date (deadline)
            - lcd_days_remaining: Days until deadline
            - is_important: Boolean importance flag from IsImportant column
            - product_code: Task/Process description
        """
        if not self.connection:
            self.connect()
            
        query = """
        SELECT 
            t.TxnId_i,
            t.SessionKey_v,
            t.DocRef_v,
            t.ItemId_i,
            t.CustId_i,
            t.TargetDate_dd,
            t.JoQty_d,
            t.QtyDone_d,
            t.DocStatus_c,
            t.Void_c,
            t.IsImportant,
            p.ProcessId_i,
            p.RowId_i,
            p.Task_v,
            p.ProcessDescr_v,
            p.Machine_v,
            p.CapQty_d,
            p.CapMin_d,
            p.SetupTime_d,
            p.LeadTime_d,
            p.ManCount_i,
            p.DifficultyLevel_i,
            p.QtyStatus_c
        FROM tbl_jo_txn t
        LEFT JOIN tbl_jo_process p ON t.TxnId_i = p.TxnId_i
        WHERE t.DocStatus_c = 'DF' 
          AND t.Void_c = '0'
          AND p.Void_c = '0'
          AND p.QtyStatus_c = 'DF'
          AND t.QtyDone_d < t.JoQty_d
        ORDER BY t.TargetDate_dd, t.TxnId_i, p.RowId_i
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                raw_jobs = cursor.fetchall()
                
            # No longer need machine type mapping since Machine_v contains IDs directly
                
            # Process results
            processed_jobs = []
            
            for job in raw_jobs:
                # Create unique job_id using DocRef and Task
                job_id = f"{job['DocRef_v']}_{job['Task_v']}"
                
                # Use DocRef as family_id (groups related jobs)
                family_id = job['DocRef_v']
                
                # Calculate days remaining
                if job['TargetDate_dd']:
                    days_remaining = (job['TargetDate_dd'] - datetime.now().date()).days
                else:
                    days_remaining = 30  # Default if no target date
                    
                # Parse required machines from Machine_v field
                # Machine_v contains machine IDs that must ALL be used simultaneously
                required_machines = []
                if job['Machine_v']:
                    # Machine_v contains comma-separated machine IDs
                    machine_ids = [m.strip() for m in str(job['Machine_v']).split(',')]
                    # Convert to integers
                    for machine_id in machine_ids:
                        try:
                            required_machines.append(int(machine_id))
                        except ValueError:
                            logger.warning(f"Invalid machine ID: {machine_id} in job {job_id}")
                
                # If no machines specified, this job cannot be scheduled
                if not required_machines:
                    logger.warning(f"Job {job_id} has no assigned machines, skipping")
                    continue
                    
                # Use IsImportant flag from tbl_jo_txn
                is_important = bool(job.get('IsImportant', 0))
                
                # Calculate processing time based on capacity
                # When CapMin_d = 1 and CapQty_d > 0: capacity is per minute
                if job.get('CapMin_d') == 1 and job.get('CapQty_d', 0) > 0:
                    # Hourly capacity = CapQty_d * 60 (convert per-minute to per-hour)
                    hourly_capacity = float(job['CapQty_d']) * 60
                    # Hours needed = JoQty_d / hourly_capacity
                    hours_needed = float(job.get('JoQty_d', 0)) / hourly_capacity
                    processing_time = hours_needed
                else:
                    # Default to 1 hour if no capacity data
                    processing_time = 1.0
                
                # Add setup time (convert minutes to hours)
                if job.get('SetupTime_d'):
                    processing_time += float(job['SetupTime_d']) / 60
                    
                processed_job = {
                    'job_id': job_id,
                    'family_id': family_id,
                    'sequence': job['RowId_i'],  # Process sequence within job
                    'required_machines': required_machines,  # ALL machines needed simultaneously
                    'processing_time': processing_time,
                    'lcd_date': job['TargetDate_dd'].isoformat() if job['TargetDate_dd'] else None,
                    'lcd_days_remaining': days_remaining,
                    'is_important': is_important,
                    'product_code': job.get('Task_v', ''),
                    'process_description': job.get('ProcessDescr_v', ''),
                    'quantity': float(job.get('JoQty_d', 0) or 0),
                    'quantity_done': float(job.get('QtyDone_d', 0) or 0),
                    'txn_id': job['TxnId_i'],
                    'process_id': job['ProcessId_i'],
                    'difficulty_level': job.get('DifficultyLevel_i', 1)
                }
                processed_jobs.append(processed_job)
                
            logger.info(f"Fetched {len(processed_jobs)} pending jobs from database")
            return processed_jobs
            
        except Exception as e:
            logger.error(f"Error fetching pending jobs: {e}")
            raise
            
    def fetch_machines(self) -> List[Dict[str, Any]]:
        """
        Fetch all active machines from database.
        
        Returns:
            List of machine dictionaries with fields:
            - machine_id: Unique machine identifier (0-based index for environment)
            - machine_name: Machine name from database
            - machine_type_id: Machine type for compatibility checking
            - db_machine_id: Original database ID
        """
        if not self.connection:
            self.connect()
            
        query = """
        SELECT 
            MachineId_i,
            MachineName_v,
            MachineDescr_v,
            MachinetypeId_i,
            DivisionId_i,
            LocId_i,
            Status_i
        FROM tbl_machine
        WHERE Status_i = 1
        ORDER BY MachineId_i
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                machines = cursor.fetchall()
                
            # Process results with 0-based indexing for environment
            processed_machines = []
            for idx, machine in enumerate(machines):
                processed_machine = {
                    'machine_id': idx,  # 0-based index for environment
                    'machine_name': machine['MachineName_v'] or f"Machine_{machine['MachineId_i']}",
                    'machine_type_id': machine['MachinetypeId_i'] or 1,  # Default type 1 if null
                    'db_machine_id': machine['MachineId_i'],  # Keep original ID
                    'division_id': machine['DivisionId_i'],
                    'location_id': machine['LocId_i'],
                    'description': machine['MachineDescr_v']
                }
                processed_machines.append(processed_machine)
                
            logger.info(f"Fetched {len(processed_machines)} machines from database")
            return processed_machines
            
        except Exception as e:
            logger.error(f"Error fetching machines: {e}")
            raise
            
    def fetch_working_hours(self) -> Dict[str, Any]:
        """
        Fetch working hours configuration from ai_arrangable_hour table.
        
        Returns:
            Dictionary with working hours configuration
        """
        if not self.connection:
            self.connect()
            
        query = """
        SELECT 
            arrange_day,
            start_time,
            end_time,
            is_working
        FROM ai_arrangable_hour
        WHERE is_working = 1
        ORDER BY arrange_day
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                hours = cursor.fetchall()
                
            # Map arrange_day to day names
            day_map = {
                1: 'monday',
                2: 'tuesday', 
                3: 'wednesday',
                4: 'thursday',
                5: 'friday',
                6: 'saturday',
                7: 'sunday'
            }
            
            working_hours = {}
            for hour in hours:
                day_name = day_map.get(hour['arrange_day'], f"day_{hour['arrange_day']}")
                working_hours[day_name] = {
                    'start': hour['start_time'].seconds // 3600,  # Convert to hours
                    'end': hour['end_time'].seconds // 3600
                }
                
            # Add break times
            break_query = """
            SELECT 
                name,
                start_time,
                end_time,
                duration_minutes
            FROM ai_breaktimes
            WHERE is_active = 1
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(break_query)
                breaks = cursor.fetchall()
                
            working_hours['breaks'] = []
            for break_time in breaks:
                working_hours['breaks'].append({
                    'name': break_time['name'],
                    'start': break_time['start_time'].seconds // 3600,
                    'end': break_time['end_time'].seconds // 3600
                })
                
            logger.info(f"Fetched working hours configuration")
            return working_hours
            
        except Exception as e:
            logger.error(f"Error fetching working hours: {e}")
            # Return default if error
            return self._get_default_working_hours()
    
    def _get_default_working_hours(self) -> Dict[str, Any]:
        """Get default working hours configuration."""
        return {
            'monday': {'start': 8, 'end': 18},
            'tuesday': {'start': 8, 'end': 18},
            'wednesday': {'start': 8, 'end': 18},
            'thursday': {'start': 8, 'end': 18},
            'friday': {'start': 8, 'end': 18},
            'saturday': {'start': 0, 'end': 0},
            'sunday': {'start': 0, 'end': 0},
            'breaks': [
                {'name': 'Lunch', 'start': 12, 'end': 13}
            ]
        }
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connect()
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        finally:
            self.disconnect()
            
            
    def get_job_count_summary(self) -> Dict[str, int]:
        """
        Get summary of job counts for debugging.
        
        Returns:
            Dictionary with job count statistics
        """
        if not self.connection:
            self.connect()
            
        try:
            summary = {}
            
            # Total pending jobs
            query1 = """
            SELECT COUNT(DISTINCT t.TxnId_i) as txn_count,
                   COUNT(*) as process_count
            FROM tbl_jo_txn t
            LEFT JOIN tbl_jo_process p ON t.TxnId_i = p.TxnId_i
            WHERE t.DocStatus_c = 'DF' 
              AND t.Void_c = '0'
              AND p.Void_c = '0'
              AND p.QtyStatus_c = 'DF'
              AND t.QtyDone_d < t.JoQty_d
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query1)
                result = cursor.fetchone()
                summary['unique_transactions'] = result['txn_count']
                summary['total_processes'] = result['process_count']
                
            # Jobs by DocRef pattern
            query2 = """
            SELECT 
                SUBSTRING(t.DocRef_v, 1, 4) as prefix,
                COUNT(DISTINCT t.TxnId_i) as count
            FROM tbl_jo_txn t
            WHERE t.DocStatus_c = 'DF' 
              AND t.Void_c = '0'
            GROUP BY SUBSTRING(t.DocRef_v, 1, 4)
            ORDER BY count DESC
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query2)
                prefixes = cursor.fetchall()
                summary['job_prefixes'] = {row['prefix']: row['count'] for row in prefixes}
                
            return summary
            
        except Exception as e:
            logger.error(f"Error getting job summary: {e}")
            return {}