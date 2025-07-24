"""
Comprehensive tests for data pipeline components (db_connector, data_loader)
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.db_connector import DBConnector
from src.data.data_loader import DataLoader


class TestDatabaseConnector:
    """Test DatabaseConnector functionality."""
    
    def test_processing_time_calculation(self):
        """Test processing time calculation with capacity formula."""
        print("\n=== Testing Processing Time Calculation ===")
        
        # Test case 1: Capacity-based calculation
        job1 = {
            'JoQty_d': 600,
            'CapQty_d': 10,
            'CapMin_d': 1,
            'SetupTime_d': 30,
            'LeadTime_d': 0
        }
        
        # Expected: (600 / (10 * 60)) + (30 / 60) = 1 + 0.5 = 1.5 hours
        processing_time = DBConnector._calculate_processing_time(job1)
        expected = 1.5
        assert abs(processing_time - expected) < 0.001, f"Expected {expected}, got {processing_time}"
        print(f"✓ Capacity-based calculation: {processing_time} hours (expected {expected})")
        
        # Test case 2: Direct time calculation (CapMin_d = 0)
        job2 = {
            'JoQty_d': 100,
            'CapQty_d': 10,
            'CapMin_d': 0,
            'SetupTime_d': 60,
            'LeadTime_d': 120  # 2 hours
        }
        
        # Expected: (120 / 60) + (60 / 60) = 2 + 1 = 3 hours
        processing_time = DBConnector._calculate_processing_time(job2)
        expected = 3.0
        assert abs(processing_time - expected) < 0.001, f"Expected {expected}, got {processing_time}"
        print(f"✓ Direct time calculation: {processing_time} hours (expected {expected})")
        
        # Test case 3: No capacity data
        job3 = {
            'JoQty_d': 100,
            'CapQty_d': 0,  # No capacity
            'CapMin_d': 1,
            'SetupTime_d': 45,
            'LeadTime_d': 90
        }
        
        # Expected: (90 / 60) + (45 / 60) = 1.5 + 0.75 = 2.25 hours
        processing_time = DBConnector._calculate_processing_time(job3)
        expected = 2.25
        assert abs(processing_time - expected) < 0.001, f"Expected {expected}, got {processing_time}"
        print(f"✓ No capacity fallback: {processing_time} hours (expected {expected})")
        
    def test_machine_parsing(self):
        """Test parsing of machine IDs from Machine_v field."""
        print("\n=== Testing Machine Parsing ===")
        
        # Test case 1: Single machine
        machines_str = "45"
        machines = DBConnector._parse_machines(machines_str)
        assert machines == [45], f"Expected [45], got {machines}"
        print(f"✓ Single machine: '{machines_str}' -> {machines}")
        
        # Test case 2: Multiple machines
        machines_str = "57,64,65,66,74"
        machines = DBConnector._parse_machines(machines_str)
        expected = [57, 64, 65, 66, 74]
        assert machines == expected, f"Expected {expected}, got {machines}"
        print(f"✓ Multiple machines: '{machines_str}' -> {machines}")
        
        # Test case 3: Empty string
        machines_str = ""
        machines = DBConnector._parse_machines(machines_str)
        assert machines == [], f"Expected [], got {machines}"
        print(f"✓ Empty string: '{machines_str}' -> {machines}")
        
        # Test case 4: None value
        machines = DBConnector._parse_machines(None)
        assert machines == [], f"Expected [], got {machines}"
        print(f"✓ None value: None -> {machines}")
        
        # Test case 5: Whitespace handling
        machines_str = " 12 , 34 , 56 "
        machines = DBConnector._parse_machines(machines_str)
        expected = [12, 34, 56]
        assert machines == expected, f"Expected {expected}, got {machines}"
        print(f"✓ Whitespace handling: '{machines_str}' -> {machines}")


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def setup_method(self):
        """Create test snapshot data."""
        self.test_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "jobs": [
                {
                    "JoId_v": "JOAW001",
                    "Task_v": "Task1",
                    "JoQty_d": 100,
                    "IsImportant": 1,
                    "TargetDate_dd": 5,
                    "processing_time": 2.5,
                    "required_machines": [1, 2]
                },
                {
                    "JoId_v": "JOST002",
                    "Task_v": "Task2", 
                    "JoQty_d": 200,
                    "IsImportant": 0,
                    "TargetDate_dd": 10,
                    "processing_time": 3.0,
                    "required_machines": [3]
                }
            ],
            "machines": [
                {"MachineId_i": 1, "MachineName_v": "CM01"},
                {"MachineId_i": 2, "MachineName_v": "CM02"},
                {"MachineId_i": 3, "MachineName_v": "CL01"}
            ],
            "working_hours": {
                "start_hour": 8,
                "end_hour": 17,
                "days_per_week": 5
            }
        }
        
        # Save test snapshot
        self.test_file = "/Users/carrickcheah/Project/ppo/app_2/tests/test_snapshot.json"
        with open(self.test_file, 'w') as f:
            json.dump(self.test_snapshot, f)
            
    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def test_load_snapshot(self):
        """Test loading data from snapshot."""
        print("\n=== Testing Snapshot Loading ===")
        
        config = {
            "source": "snapshot",
            "snapshot_path": self.test_file
        }
        
        loader = DataLoader(config)
        
        # Test job loading
        jobs = loader.load_jobs()
        assert len(jobs) == 2, f"Expected 2 jobs, got {len(jobs)}"
        assert jobs[0]['JoId_v'] == "JOAW001"
        assert jobs[0]['processing_time'] == 2.5
        assert jobs[0]['required_machines'] == [1, 2]
        print(f"✓ Loaded {len(jobs)} jobs correctly")
        
        # Test machine loading
        machines = loader.load_machines()
        assert len(machines) == 3, f"Expected 3 machines, got {len(machines)}"
        assert machines[0]['MachineId_i'] == 1
        assert machines[0]['MachineName_v'] == "CM01"
        print(f"✓ Loaded {len(machines)} machines correctly")
        
        # Test working hours loading
        working_hours = loader.load_working_hours()
        assert working_hours['start_hour'] == 8
        assert working_hours['end_hour'] == 17
        assert working_hours['days_per_week'] == 5
        print(f"✓ Loaded working hours correctly")
        
    def test_job_filtering(self):
        """Test job filtering with max_jobs parameter."""
        print("\n=== Testing Job Filtering ===")
        
        config = {
            "source": "snapshot",
            "snapshot_path": self.test_file,
            "max_jobs": 1
        }
        
        loader = DataLoader(config)
        jobs = loader.load_jobs()
        
        assert len(jobs) == 1, f"Expected 1 job, got {len(jobs)}"
        assert jobs[0]['JoId_v'] == "JOAW001"  # Should get first job
        print(f"✓ Job filtering with max_jobs=1 works correctly")
        
    def test_machine_filtering(self):
        """Test machine filtering with max_machines parameter."""
        print("\n=== Testing Machine Filtering ===")
        
        config = {
            "source": "snapshot",
            "snapshot_path": self.test_file,
            "max_machines": 2
        }
        
        loader = DataLoader(config)
        machines = loader.load_machines()
        
        assert len(machines) == 2, f"Expected 2 machines, got {len(machines)}"
        assert machines[0]['MachineId_i'] == 1
        assert machines[1]['MachineId_i'] == 2
        print(f"✓ Machine filtering with max_machines=2 works correctly")


def run_data_pipeline_tests():
    """Run all data pipeline tests."""
    print("\n" + "="*60)
    print("DATA PIPELINE TESTS")
    print("="*60)
    
    # Test DatabaseConnector
    db_tests = TestDatabaseConnector()
    db_tests.test_processing_time_calculation()
    db_tests.test_machine_parsing()
    
    # Test DataLoader
    loader_tests = TestDataLoader()
    loader_tests.setup_method()
    try:
        loader_tests.test_load_snapshot()
        loader_tests.test_job_filtering()
        loader_tests.test_machine_filtering()
    finally:
        loader_tests.teardown_method()
        
    print("\n✅ All data pipeline tests passed!")
    

if __name__ == "__main__":
    run_data_pipeline_tests()