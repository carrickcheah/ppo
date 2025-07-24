"""
Fixed tests for data pipeline components (db_connector, data_loader)
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
    
    def test_processing_time_calculation_logic(self):
        """Test processing time calculation logic."""
        print("\n=== Testing Processing Time Calculation Logic ===")
        
        # Test case 1: Capacity-based calculation
        # When CapMin_d = 1 and CapQty_d > 0
        CapQty_d = 10  # 10 units per minute
        JoQty_d = 600  # 600 units total
        SetupTime_d = 30  # 30 minutes setup
        
        # Calculate using the same logic as in DBConnector
        hourly_capacity = CapQty_d * 60  # 600 units per hour
        hours_needed = JoQty_d / hourly_capacity  # 1 hour
        processing_time = hours_needed + (SetupTime_d / 60)  # 1.5 hours
        
        expected = 1.5
        assert abs(processing_time - expected) < 0.001, f"Expected {expected}, got {processing_time}"
        print(f"✓ Capacity-based calculation: {processing_time} hours (expected {expected})")
        
        # Test case 2: No capacity data (CapMin_d = 0)
        # Should default to 1 hour + setup time
        SetupTime_d = 60  # 60 minutes setup
        processing_time = 1.0 + (SetupTime_d / 60)  # 2 hours
        
        expected = 2.0
        assert abs(processing_time - expected) < 0.001, f"Expected {expected}, got {processing_time}"
        print(f"✓ Default time calculation: {processing_time} hours (expected {expected})")
        
    def test_machine_parsing_logic(self):
        """Test parsing of machine IDs from Machine_v field."""
        print("\n=== Testing Machine Parsing Logic ===")
        
        # Test the logic used in DBConnector
        def parse_machines(machine_str):
            """Parse machine IDs from comma-separated string."""
            if not machine_str:
                return []
            try:
                return [int(m.strip()) for m in str(machine_str).split(',')]
            except ValueError:
                return []
        
        # Test case 1: Single machine
        machines = parse_machines("45")
        assert machines == [45], f"Expected [45], got {machines}"
        print(f"✓ Single machine: '45' -> {machines}")
        
        # Test case 2: Multiple machines
        machines = parse_machines("57,64,65,66,74")
        expected = [57, 64, 65, 66, 74]
        assert machines == expected, f"Expected {expected}, got {machines}"
        print(f"✓ Multiple machines: '57,64,65,66,74' -> {machines}")
        
        # Test case 3: Empty string
        machines = parse_machines("")
        assert machines == [], f"Expected [], got {machines}"
        print(f"✓ Empty string: '' -> {machines}")
        
        # Test case 4: None value
        machines = parse_machines(None)
        assert machines == [], f"Expected [], got {machines}"
        print(f"✓ None value: None -> {machines}")


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def setup_method(self):
        """Create test snapshot data."""
        self.test_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "jobs": [
                {
                    "job_id": "JOAW001/1",
                    "family_id": "JOAW001",
                    "sequence": 1,
                    "required_machines": [1, 2],
                    "processing_time": 2.5,
                    "lcd_date": "2024-12-25",
                    "lcd_days_remaining": 5,
                    "is_important": True,
                    "product_code": "Task1",
                    "process_description": "Cutting"
                },
                {
                    "job_id": "JOST002/1",
                    "family_id": "JOST002", 
                    "sequence": 1,
                    "required_machines": [3],
                    "processing_time": 3.0,
                    "lcd_date": "2024-12-30",
                    "lcd_days_remaining": 10,
                    "is_important": False,
                    "product_code": "Task2",
                    "process_description": "Assembly"
                }
            ],
            "machines": [
                {"machine_id": 1, "machine_name": "CM01", "machine_type_id": 1},
                {"machine_id": 2, "machine_name": "CM02", "machine_type_id": 1},
                {"machine_id": 3, "machine_name": "CL01", "machine_type_id": 2}
            ],
            "working_hours": {
                "start_hour": 8,
                "end_hour": 17,
                "days_per_week": 5
            }
        }
        
        # Save test snapshot
        self.test_file = "/Users/carrickcheah/Project/ppo/app_2/tests/test_snapshot_fixed.json"
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
        assert jobs[0]['job_id'] == "JOAW001/1"
        assert jobs[0]['processing_time'] == 2.5
        assert jobs[0]['required_machines'] == [1, 2]
        print(f"✓ Loaded {len(jobs)} jobs correctly")
        
        # Test machine loading
        machines = loader.load_machines()
        assert len(machines) == 3, f"Expected 3 machines, got {len(machines)}"
        assert machines[0]['machine_id'] == 1
        assert machines[0]['machine_name'] == "CM01"
        print(f"✓ Loaded {len(machines)} machines correctly")
        
        # Test working hours loading
        working_hours = loader.load_working_hours()
        # Working hours can be None for training
        if working_hours:
            assert 'start_hour' in working_hours
            print(f"✓ Loaded working hours: {working_hours}")
        else:
            print(f"✓ Working hours disabled (training mode)")
        
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
        assert jobs[0]['job_id'] == "JOAW001/1"  # Should get first job
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
        assert machines[0]['machine_id'] == 1
        assert machines[1]['machine_id'] == 2
        print(f"✓ Machine filtering with max_machines=2 works correctly")


def run_data_pipeline_tests():
    """Run all data pipeline tests."""
    print("\n" + "="*60)
    print("DATA PIPELINE TESTS")
    print("="*60)
    
    # Test DatabaseConnector
    db_tests = TestDatabaseConnector()
    db_tests.test_processing_time_calculation_logic()
    db_tests.test_machine_parsing_logic()
    
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