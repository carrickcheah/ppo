"""
Verify the system is ready for training with real production data
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("TRAINING READINESS CHECK")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check 1: Production data exists
print("\n1. Checking production data...")
snapshot_path = "data/real_production_snapshot.json"
if os.path.exists(snapshot_path):
    with open(snapshot_path, 'r') as f:
        data = json.load(f)
    n_jobs = len(data.get('jobs', []))
    n_machines = len(data.get('machines', []))
    print(f"✓ Production snapshot exists: {n_jobs} jobs, {n_machines} machines")
else:
    print("✗ No production snapshot found. Run data ingestion first.")
    print("  Run: uv run python src/data_ingestion/ingest_data.py --output data/real_production_snapshot.json")
    sys.exit(1)

# Check 2: Configurations exist
print("\n2. Checking configurations...")
configs = ['configs/training.yaml', 'configs/environment.yaml']
for config in configs:
    if os.path.exists(config):
        print(f"✓ {config} exists")
    else:
        print(f"✗ {config} missing")
        sys.exit(1)

# Check 3: Test with small subset
print("\n3. Testing with production data subset...")
try:
    from src.data.data_loader import DataLoader
    from src.environment.scheduling_game_env import SchedulingGameEnv
    
    # Load subset
    loader = DataLoader({
        "source": "snapshot",
        "snapshot_path": snapshot_path,
        "max_jobs": 10,
        "max_machines": 5
    })
    
    jobs = loader.load_jobs()
    machines = loader.load_machines()
    
    # Create environment
    env = SchedulingGameEnv(jobs, machines, None, {})
    obs, info = env.reset()
    
    # Check action space
    mask = env.get_action_mask()
    valid_actions = np.sum(mask)
    
    print(f"✓ Environment test successful")
    print(f"  Jobs: {len(jobs)}")
    print(f"  Machines: {len(machines)}")
    print(f"  Valid actions: {valid_actions}/{len(mask)}")
    
    # Try one step if possible
    if valid_actions > 0:
        valid_idx = np.where(mask)[0][0]
        job_idx = valid_idx // len(machines)
        machine_idx = valid_idx % len(machines)
        action = np.array([job_idx, machine_idx])
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Test step successful, reward: {reward}")
    
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Model components
print("\n4. Checking model components...")
try:
    from phase2.ppo_scheduler import PPOScheduler
    from phase2.curriculum import CurriculumManager
    from phase2.rollout_buffer import RolloutBuffer
    
    print("✓ All model components importable")
    
except Exception as e:
    print(f"✗ Model component import failed: {e}")
    sys.exit(1)

# Check 5: Training script
print("\n5. Checking training script...")
if os.path.exists("phase2/train.py"):
    print("✓ Training script exists")
else:
    print("✗ Training script missing")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✅ SYSTEM READY FOR TRAINING!")
print("="*60)
print(f"\nProduction data: {n_jobs} jobs, {n_machines} machines")
print("All components verified and working")
print("\nTo start training with curriculum learning:")
print("  uv run python phase2/train.py --config configs/training.yaml")
print("\nOr use the quick start script:")
print("  uv run python start_training.py")
print("\nExpected training time: 2-4 hours")
print("The model will progressively learn from toy → production scale")