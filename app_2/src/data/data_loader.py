"""
Data Loader for Scheduling Game

Loads job and machine data from various sources:
- Direct from MariaDB database
- From JSON snapshots (for reproducibility)
- From test files

No validation or business logic - just loading raw data.
"""

import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from .db_connector import DBConnector

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads scheduling data from various sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            config: Data loading configuration
        """
        self.config = config
        self.source = config.get('source', 'database')  # database, snapshot, or test
        self.db_connector = None
        
        if self.source == 'database':
            self.db_connector = DBConnector(config.get('database', {}))
            
    def load_jobs(self) -> List[Dict[str, Any]]:
        """
        Load job data based on configured source.
        
        Returns:
            List of job dictionaries
        """
        if self.source == 'database':
            return self._load_jobs_from_database()
        elif self.source == 'snapshot':
            return self._load_jobs_from_snapshot()
        elif self.source == 'test':
            return self._load_test_jobs()
        else:
            raise ValueError(f"Unknown data source: {self.source}")
            
    def load_machines(self) -> List[Dict[str, Any]]:
        """
        Load machine data based on configured source.
        
        Returns:
            List of machine dictionaries
        """
        if self.source == 'database':
            return self._load_machines_from_database()
        elif self.source == 'snapshot':
            return self._load_machines_from_snapshot()
        elif self.source == 'test':
            return self._load_test_machines()
        else:
            raise ValueError(f"Unknown data source: {self.source}")
            
    def load_working_hours(self) -> Dict[str, Any]:
        """
        Load working hours configuration.
        
        Returns:
            Working hours dictionary
        """
        if self.source == 'database':
            return self.db_connector.fetch_working_hours()
        else:
            # Use default for all sources
            return self._get_default_working_hours()
            
    def _load_jobs_from_database(self) -> List[Dict[str, Any]]:
        """Load jobs directly from MariaDB."""
        if not self.db_connector:
            self.db_connector = DBConnector(self.config.get('database', {}))
            
        try:
            jobs = self.db_connector.fetch_pending_jobs()
            logger.info(f"Loaded {len(jobs)} jobs from database")
            return jobs
        finally:
            if self.db_connector:
                self.db_connector.disconnect()
                
    def _load_machines_from_database(self) -> List[Dict[str, Any]]:
        """Load machines directly from MariaDB."""
        if not self.db_connector:
            self.db_connector = DBConnector(self.config.get('database', {}))
            
        try:
            machines = self.db_connector.fetch_machines()
            
            # Apply machine limit if specified
            max_machines = self.config.get('max_machines')
            if max_machines and len(machines) > max_machines:
                machines = machines[:max_machines]
                logger.info(f"Limited to {max_machines} machines")
                
            logger.info(f"Loaded {len(machines)} machines from database")
            return machines
        finally:
            if self.db_connector:
                self.db_connector.disconnect()
                
    def _load_jobs_from_snapshot(self) -> List[Dict[str, Any]]:
        """Load jobs from JSON snapshot file."""
        snapshot_path = Path(self.config.get('snapshot_path', 'data/production_snapshot.json'))
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
            
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
            
        # Check for both possible keys
        jobs = data.get('jobs', data.get('pending_jobs', []))
        
        # Apply job limit if specified
        max_jobs = self.config.get('max_jobs')
        if max_jobs and len(jobs) > max_jobs:
            jobs = jobs[:max_jobs]
            logger.info(f"Limited to {max_jobs} jobs")
            
        logger.info(f"Loaded {len(jobs)} jobs from snapshot: {snapshot_path}")
        return jobs
        
    def _load_machines_from_snapshot(self) -> List[Dict[str, Any]]:
        """Load machines from JSON snapshot file."""
        snapshot_path = Path(self.config.get('snapshot_path', 'data/production_snapshot.json'))
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
            
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
            
        machines = data.get('machines', [])
        
        # Apply machine limit if specified
        max_machines = self.config.get('max_machines')
        if max_machines and len(machines) > max_machines:
            machines = machines[:max_machines]
            logger.info(f"Limited to {max_machines} machines")
            
        logger.info(f"Loaded {len(machines)} machines from snapshot: {snapshot_path}")
        return machines
        
    def _load_test_jobs(self) -> List[Dict[str, Any]]:
        """Load small test dataset for development."""
        # Create simple test jobs
        test_jobs = []
        
        # Family 1: 3 sequential jobs (important)
        for seq in range(1, 4):
            test_jobs.append({
                'job_id': f'TEST001_{seq}/3',
                'family_id': 'TEST001',
                'sequence': seq,
                'machine_types': [1, 2, 3],
                'processing_time': 2.0,
                'lcd_date': datetime.now().isoformat(),
                'lcd_days_remaining': 5,
                'is_important': True,
                'product_code': 'CF001'
            })
            
        # Family 2: 2 sequential jobs (not important)
        for seq in range(1, 3):
            test_jobs.append({
                'job_id': f'TEST002_{seq}/2',
                'family_id': 'TEST002',
                'sequence': seq,
                'machine_types': [2, 3, 4],
                'processing_time': 1.5,
                'lcd_date': datetime.now().isoformat(),
                'lcd_days_remaining': 10,
                'is_important': False,
                'product_code': 'CP002'
            })
            
        # Single job (no family)
        test_jobs.append({
            'job_id': 'TEST003',
            'family_id': 'TEST003',
            'sequence': 1,
            'machine_types': [1, 2, 3, 4, 5],
            'processing_time': 3.0,
            'lcd_date': datetime.now().isoformat(),
            'lcd_days_remaining': 3,
            'is_important': True,
            'product_code': 'CD003'
        })
        
        logger.info(f"Created {len(test_jobs)} test jobs")
        return test_jobs
        
    def _load_test_machines(self) -> List[Dict[str, Any]]:
        """Load small test machine set."""
        test_machines = []
        
        # Create 5 test machines with different types
        machine_names = ['M01', 'M02', 'M03', 'M04', 'M05']
        machine_types = [1, 2, 3, 4, 5]
        
        for idx, (name, mtype) in enumerate(zip(machine_names, machine_types)):
            test_machines.append({
                'machine_id': idx,
                'machine_name': name,
                'machine_type_id': mtype,
                'db_machine_id': idx + 1
            })
            
        logger.info(f"Created {len(test_machines)} test machines")
        return test_machines
        
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
                {'start': 12, 'end': 13}
            ]
        }
        
    def save_snapshot(
        self,
        jobs: List[Dict[str, Any]],
        machines: List[Dict[str, Any]],
        filepath: Union[str, Path]
    ):
        """
        Save current data as snapshot for reproducibility.
        
        Args:
            jobs: List of jobs
            machines: List of machines
            filepath: Path to save snapshot
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'pending_jobs': jobs,
            'machines': machines,
            'metadata': {
                'n_jobs': len(jobs),
                'n_machines': len(machines),
                'source': self.source
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
            
        logger.info(f"Saved snapshot to {filepath}")