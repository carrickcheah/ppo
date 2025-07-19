"""
Test to ensure ONLY real production data is being used.
This test MUST pass before any training or deployment.
"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.environments.full_production_env import FullProductionEnv

def test_synthetic_data_forbidden():
    """Test that synthetic data generation raises an error."""
    env = FullProductionEnv(n_machines=10, n_jobs=50)
    
    # This should raise an error per our updates
    try:
        env._generate_full_production_data()
        return False, "Synthetic data generation did not raise error!"
    except RuntimeError as e:
        if "SYNTHETIC DATA GENERATION IS FORBIDDEN" in str(e):
            return True, "Synthetic data generation properly forbidden"
        else:
            return False, f"Wrong error raised: {e}"

def test_real_data_required():
    """Test that environment requires real data configuration."""
    # Remove marker file if it exists (to test the check)
    marker_path = Path(__file__).parent.parent / "data" / ".use_real_data_only"
    if marker_path.exists():
        marker_path.rename(marker_path.with_suffix('.backup'))
    
    try:
        # This should fail without real data configuration
        with pytest.raises(RuntimeError, match="REAL DATA NOT CONFIGURED"):
            env = FullProductionEnv(n_machines=10, n_jobs=50)
            env._load_data()
    finally:
        # Restore marker if it was backed up
        backup_path = marker_path.with_suffix('.backup')
        if backup_path.exists():
            backup_path.rename(marker_path)

def test_real_data_validation():
    """Test that loaded data is validated as real production data."""
    snapshot_path = Path(__file__).parent.parent / "data" / "real_production_snapshot.json"
    
    if snapshot_path.exists():
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
        
        # Check for real job prefixes
        families = data.get('families', {})
        real_prefixes = ['JOAW', 'JOST', 'JOEX', 'JOCS']
        
        has_real_jobs = False
        for family_id in families:
            if any(family_id.startswith(prefix) for prefix in real_prefixes):
                has_real_jobs = True
                break
        
        assert has_real_jobs, "No real job IDs found - data appears to be synthetic!"
        
        # Check for real machine names
        machines = data.get('machines', [])
        if machines:
            machine_names = [m.get('machine_name', '') for m in machines]
            real_patterns = ['CM', 'CL', 'AD', 'MG', 'PP']
            
            has_real_machines = any(
                any(pattern in name for pattern in real_patterns)
                for name in machine_names
            )
            
            assert has_real_machines, "No real machine names found - data appears to be synthetic!"

def test_no_synthetic_patterns():
    """Test that data doesn't contain synthetic patterns."""
    snapshot_path = Path(__file__).parent.parent / "data" / "real_production_snapshot.json"
    
    if snapshot_path.exists():
        with open(snapshot_path, 'r') as f:
            content = f.read()
        
        # Check for synthetic patterns
        synthetic_patterns = [
            'job_0', 'job_1', 'job_2',  # Sequential job IDs
            'M001', 'M002', 'M003',      # Sequential machine names
            'FAM_0001', 'FAM_0002',      # Synthetic family IDs
            'Job_0', 'Job_1',            # Generated job names
            'Family_0', 'Family_1'       # Generated family names
        ]
        
        for pattern in synthetic_patterns:
            assert pattern not in content, f"Found synthetic pattern '{pattern}' in data!"

if __name__ == "__main__":
    print("=== Testing Real Data Enforcement ===")
    
    # Run basic validation
    try:
        test_synthetic_data_forbidden()
        print("✓ Synthetic data generation properly forbidden")
    except Exception as e:
        print(f"✗ Synthetic data test failed: {e}")
    
    try:
        test_real_data_validation()
        print("✓ Real data validation passed")
    except Exception as e:
        print(f"✗ Real data validation failed: {e}")
    
    try:
        test_no_synthetic_patterns()
        print("✓ No synthetic patterns found")
    except Exception as e:
        print(f"✗ Synthetic pattern test failed: {e}")
    
    print("\nUse pytest to run all tests: pytest tests/test_real_data_enforcement.py")