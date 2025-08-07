#!/usr/bin/env python
"""
Monitor the 1M training progress and test when ready
"""

import os
import time
import subprocess

def check_training_status():
    """Check if training is still running."""
    result = subprocess.run(
        ["ps", "aux"], 
        capture_output=True, 
        text=True
    )
    return "train_sb3_1million.py" in result.stdout

def get_latest_checkpoint():
    """Find the latest checkpoint."""
    checkpoint_dir = "checkpoints/sb3_1million"
    if not os.path.exists(checkpoint_dir):
        return None
    
    files = os.listdir(checkpoint_dir)
    checkpoints = [f for f in files if f.startswith("checkpoint_") or f == "best_model.zip"]
    
    if not checkpoints:
        return None
    
    # Get the latest by modification time
    latest = max(
        [os.path.join(checkpoint_dir, f) for f in checkpoints],
        key=os.path.getmtime
    )
    return latest

def get_file_size_mb(filepath):
    """Get file size in MB."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0

def main():
    """Monitor training progress."""
    
    print("="*80)
    print("📊 MONITORING 1 MILLION STEP TRAINING")
    print("="*80)
    
    start_time = time.time()
    last_checkpoint = None
    
    while True:
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        print(f"\n⏱️  Time elapsed: {elapsed_min:.1f} minutes")
        
        # Check if training is running
        is_running = check_training_status()
        
        if is_running:
            print("✅ Training is running")
        else:
            print("⚠️  Training process not found")
        
        # Check for checkpoints
        latest = get_latest_checkpoint()
        
        if latest:
            size_mb = get_file_size_mb(latest)
            checkpoint_name = os.path.basename(latest)
            print(f"📁 Latest checkpoint: {checkpoint_name} ({size_mb:.1f} MB)")
            
            if latest != last_checkpoint:
                print("   🆕 New checkpoint saved!")
                last_checkpoint = latest
                
                # Extract step count from checkpoint name
                if "checkpoint_" in checkpoint_name:
                    try:
                        steps = int(checkpoint_name.split("_")[1])
                        progress = steps / 1_000_000 * 100
                        print(f"   📈 Progress: {steps:,}/1,000,000 ({progress:.1f}%)")
                        
                        # Estimate time remaining
                        if steps > 0:
                            rate = steps / elapsed
                            remaining = (1_000_000 - steps) / rate
                            print(f"   ⏱️  ETA: {remaining/60:.1f} minutes")
                    except:
                        pass
        
        # Check if final model exists
        final_model = "checkpoints/sb3_1million/final_1million_model.zip"
        if os.path.exists(final_model):
            print("\n🎉 TRAINING COMPLETE!")
            print(f"📁 Final model saved: {final_model}")
            print(f"   Size: {get_file_size_mb(final_model):.1f} MB")
            print(f"   Total time: {elapsed_min:.1f} minutes")
            
            print("\n" + "="*80)
            print("Next steps:")
            print("1. Run: python test_1million_model.py")
            print("2. Compare with all baselines")
            print("="*80)
            break
        
        # Also check for completion in log
        if os.path.exists("training_1m.log"):
            with open("training_1m.log", "r") as f:
                content = f.read()
                if "TRAINING COMPLETE!" in content or "FINAL ASSESSMENT" in content:
                    print("\n✅ Training finished! Check log for results.")
                    print("Run: tail -100 training_1m.log")
                    break
        
        if not is_running and not latest:
            print("\n❌ Training may have failed. Check training_1m.log for errors.")
            break
        
        # Wait before next check
        print("\n" + "-"*40)
        print("Checking again in 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()