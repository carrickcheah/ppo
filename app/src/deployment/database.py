"""
Database Module for PPO Scheduler API

This module handles all database connections and queries for the production scheduler.
It provides methods to fetch machines, jobs, and save scheduling results back to MariaDB.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

from .models import Machine, Job
from .settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages database connections and queries for the PPO scheduler.
    """
    
    def __init__(self):
        """Initialize database connection parameters from settings."""
        try:
            settings = get_settings()
            self.host = settings.db_host
            self.user = settings.db_user
            self.password = settings.db_password
            self.database = settings.db_name
            self.port = settings.db_port
            
            logger.info(f"Database connection initialized for {self.database}@{self.host}:{self.port}")
            logger.debug(f"Database config - Host: {self.host}, Port: {self.port}, User: {self.user}, DB: {self.database}")
        except Exception as e:
            logger.error(f"Failed to initialize database settings: {type(e).__name__}: {str(e)}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures connections are properly closed after use.
        """
        connection = None
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
            yield connection
        except pymysql.Error as e:
            logger.error(f"MySQL connection error ({e.args[0]}): {e.args[1]}")
            logger.error(f"Connection params - Host: {self.host}:{self.port}, User: {self.user}, Database: {self.database}")
            raise
        except Exception as e:
            logger.error(f"Database connection error: {type(e).__name__}: {str(e)}")
            raise
        finally:
            if connection:
                connection.close()
    
    def get_machines(self) -> List[Machine]:
        """
        Fetch all active machines from the database.
        
        Returns:
            List of Machine objects with their current status
        """
        query = """
        SELECT 
            MachineId_i as machine_id,
            MachineName_v as machine_name,
            MachinetypeId_i as machine_type,
            0 as current_load,  -- CurrentLoad_d column doesn't exist
            Status_i as status
        FROM tbl_machine
        WHERE Status_i = 1  -- Active machines only
        ORDER BY MachineId_i
        """
        
        machines = []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    for row in results:
                        machine = Machine(
                            machine_id=row['machine_id'],
                            machine_name=row['machine_name'],
                            machine_type=row['machine_type'],
                            current_load=float(row['current_load'])
                        )
                        machines.append(machine)
                    
                    logger.info(f"Loaded {len(machines)} active machines from database")
                    
        except Exception as e:
            logger.error(f"Failed to fetch machines: {str(e)}")
            raise
        
        return machines
    
    def get_pending_jobs(self, limit: Optional[int] = None) -> List[Job]:
        """
        Fetch pending jobs that need to be scheduled.
        
        Args:
            limit: Maximum number of jobs to fetch (None for all)
            
        Returns:
            List of Job objects ready for scheduling
        """
        # Use the same query structure as ingest_data.py
        query = """
        SELECT
            jot.DocRef_v AS job_id,
            jot.DocRef_v AS family_id,
            jop.RowId_i AS sequence,
            jot.TargetDate_dd AS lcd_date,
            jot.JoQty_d AS quantity,
            tm.MachinetypeId_i AS machine_type,
            jop.SetupTime_d AS setup_time,
            jop.QtyStatus_c AS status,
            CASE WHEN jot.TargetDate_dd <= DATE_ADD(CURDATE(), INTERVAL 7 DAY) THEN 'High' ELSE 'Normal' END AS importance,
            
            -- Calculate processing time in hours
            CASE 
                WHEN jop.CapMin_d = 1 AND jop.CapQty_d > 0 THEN
                    jot.JoQty_d / (jop.CapQty_d * 60)
                WHEN jop.LeadTime_d > 0 THEN
                    jop.LeadTime_d * 8  -- Convert days to hours (8 hour workday)
                ELSE 
                    2.0  -- Default 2 hours if no data
            END AS processing_time,
            
            jop.Machine_v,
            jop.Task_v AS process_code
            
        FROM tbl_jo_process AS jop
        INNER JOIN tbl_jo_txn AS jot ON jot.TxnId_i = jop.TxnId_i
        LEFT JOIN tbl_machine AS tm ON tm.MachineId_i = CAST(jop.Machine_v AS UNSIGNED)
        WHERE 
            jot.Void_c != 1
            AND jot.DocStatus_c NOT IN ('CP', 'CX')
            AND jop.QtyStatus_c != 'FF'  -- Not finished
            AND jot.TargetDate_dd > CURDATE()  -- Future deadlines only
            AND jot.TargetDate_dd <= DATE_ADD(CURDATE(), INTERVAL 30 DAY)
            AND jot.MaterialDate_dd IS NOT NULL
            AND jot.MaterialDate_dd <= CURDATE()  -- Material arrived
        ORDER BY 
            jot.TargetDate_dd ASC,  -- Earliest deadline first
            jop.TxnId_i ASC,
            jop.RowId_i ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        jobs = []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    # Convert results to Job objects
                    for row in results:
                        # Parse LCD date
                        lcd_date = row['lcd_date']
                        if isinstance(lcd_date, str):
                            lcd_date = datetime.strptime(lcd_date, '%Y-%m-%d')
                        
                        # Determine machine types based on assigned machine or process code
                        machine_types = []
                        if row['machine_type']:
                            machine_types.append(row['machine_type'])
                        else:
                            # For manual processes, allow all machine types (1-10)
                            machine_types = list(range(1, 11))
                        
                        job = Job(
                            job_id=row['job_id'],
                            family_id=row['family_id'] or row['job_id'],
                            sequence=row['sequence'] or 1,
                            processing_time=float(row['processing_time']),
                            machine_types=machine_types,
                            is_important=row['importance'] == 'High',
                            lcd_date=lcd_date,
                            setup_time=float(row['setup_time']) if row['setup_time'] else 0.3
                        )
                        jobs.append(job)
                    
                    logger.info(f"Loaded {len(jobs)} pending jobs from database")
                    
        except Exception as e:
            logger.error(f"Failed to fetch pending jobs: {str(e)}")
            raise
        
        return jobs
    
    def save_schedule(self, schedule_id: str, scheduled_jobs: List[Dict[str, Any]], 
                     metrics: Dict[str, Any]) -> bool:
        """
        Save the generated schedule back to the database.
        
        Args:
            schedule_id: Unique identifier for this schedule
            scheduled_jobs: List of scheduled job assignments
            metrics: Performance metrics for the schedule
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Start transaction
                    conn.begin()
                    
                    # Insert schedule header
                    header_query = """
                    INSERT INTO tbl_schedule_header (
                        schedule_id, created_at, makespan, completion_rate,
                        utilization, total_jobs, scheduled_jobs, algorithm
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(header_query, (
                        schedule_id,
                        datetime.now(),
                        metrics.get('makespan', 0),
                        metrics.get('completion_rate', 0),
                        metrics.get('average_utilization', 0),
                        metrics.get('total_jobs', 0),
                        metrics.get('scheduled_jobs', 0),
                        'PPO_Full_Production'
                    ))
                    
                    # Insert schedule details
                    detail_query = """
                    INSERT INTO tbl_schedule_detail (
                        schedule_id, job_id, machine_id, machine_name,
                        start_time, end_time, setup_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    for job in scheduled_jobs:
                        cursor.execute(detail_query, (
                            schedule_id,
                            job['job_id'],
                            job['machine_id'],
                            job['machine_name'],
                            job['start_datetime'],
                            job['end_datetime'],
                            job.get('setup_time_included', 0)
                        ))
                    
                    # Commit transaction
                    conn.commit()
                    logger.info(f"Schedule {schedule_id} saved successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to save schedule: {str(e)}")
            if conn:
                conn.rollback()
            return False
    
    def get_machine_availability(self, machine_id: int, 
                                start_date: datetime, 
                                end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Get availability windows for a specific machine.
        
        Args:
            machine_id: Machine ID to check
            start_date: Start of period to check
            end_date: End of period to check
            
        Returns:
            List of (start, end) tuples representing unavailable periods
        """
        query = """
        SELECT 
            start_time,
            end_time,
            reason
        FROM tbl_machine_unavailability
        WHERE 
            machine_id = %s
            AND end_time >= %s
            AND start_time <= %s
            AND status = 'Active'
        ORDER BY start_time
        """
        
        unavailable_periods = []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (machine_id, start_date, end_date))
                    results = cursor.fetchall()
                    
                    for row in results:
                        unavailable_periods.append((
                            row['start_time'],
                            row['end_time']
                        ))
                        
        except Exception as e:
            logger.error(f"Failed to fetch machine availability: {str(e)}")
            # Return empty list on error to avoid blocking scheduling
            return []
        
        return unavailable_periods
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result is not None
        except pymysql.Error as e:
            logger.error(f"Database connection test failed - MySQL Error ({e.args[0]}): {e.args[1]}")
            return False
        except Exception as e:
            logger.error(f"Database connection test failed: {type(e).__name__}: {str(e)}")
            return False


# Singleton instance
_db_connection = None


def get_database() -> DatabaseConnection:
    """
    Get the singleton database connection instance.
    
    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection