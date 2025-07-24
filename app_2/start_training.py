"""
Quick start script for training the PPO scheduling model
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Start training with production data."""
    
    # Change to app_2 directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    logger.info("="*60)
    logger.info("PPO SCHEDULING MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we have production data snapshot
    snapshot_path = "data/real_production_snapshot.json"
    if not os.path.exists(snapshot_path):
        logger.info("\n📊 First, let's fetch real production data...")
        logger.info("Running data ingestion...")
        
        # Run data ingestion
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "src/data_ingestion/ingest_data.py",
            "--output", snapshot_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("Failed to fetch production data!")
            logger.error(result.stderr)
            return
        
        logger.info("✅ Production data fetched successfully!")
    else:
        logger.info("✅ Using existing production data snapshot")
    
    # Update environment config to use snapshot
    logger.info("\n🔧 Configuring environment to use production data...")
    
    import yaml
    
    # Load environment config
    with open("configs/environment.yaml", 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Update to use snapshot
    env_config['data']['source'] = 'snapshot'
    env_config['data']['snapshot_path'] = snapshot_path
    
    # Save updated config
    with open("configs/environment.yaml", 'w') as f:
        yaml.dump(env_config, f, default_flow_style=False)
    
    logger.info("✅ Environment configured")
    
    # Create necessary directories
    logger.info("\n📁 Creating directories...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Start training
    logger.info("\n🚀 Starting PPO training with curriculum learning...")
    logger.info("This will train through progressively complex scenarios:")
    logger.info("  1. Toy (10 jobs, 5 machines)")
    logger.info("  2. Small (50 jobs, 20 machines)")
    logger.info("  3. Medium (200 jobs, 50 machines)")
    logger.info("  4. Large (500 jobs, 100 machines)")
    logger.info("  5. Production (full scale)")
    logger.info("\n⏱️  Estimated training time: 2-4 hours (depending on hardware)")
    
    # Launch training
    logger.info("\n" + "="*60)
    logger.info("LAUNCHING TRAINING...")
    logger.info("="*60)
    
    import subprocess
    subprocess.run([
        sys.executable,
        "phase2/train.py",
        "--config", "configs/training.yaml"
    ])


if __name__ == "__main__":
    main()