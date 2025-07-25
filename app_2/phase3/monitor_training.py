"""
Monitor Training Progress in Real-Time

Shows current stage, progress, and performance metrics.
"""

import os
import json
import time
from datetime import datetime, timedelta


def get_training_status():
    """Get current training status."""
    state_path = 'phase3/checkpoints/training_state.json'
    log_path = 'phase3/logs/training.log'
    
    # Load training state
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
    else:
        state = {'completed_stages': [], 'current_stage_idx': 0}
        
    # Get latest log lines
    latest_logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            latest_logs = lines[-50:]  # Last 50 lines
            
    return state, latest_logs


def parse_episode_info(log_line):
    """Extract episode information from log line."""
    if 'Episode' in log_line and 'Reward:' in log_line:
        try:
            # Extract metrics
            parts = log_line.split(' - ')
            episode_num = int(parts[1].split()[1])
            reward = float(parts[2].split()[1].rstrip(','))
            completed = int(parts[3].split()[1].rstrip(','))
            late = int(parts[4].split()[1].rstrip(','))
            utilization = float(parts[5].split()[1].rstrip('%'))
            
            return {
                'episode': episode_num,
                'reward': reward,
                'completed': completed,
                'late': late,
                'utilization': utilization
            }
        except:
            pass
    return None


def main():
    """Main monitoring loop."""
    stages = [
        'toy_easy', 'toy_normal', 'toy_hard', 'toy_multi',
        'small_balanced', 'small_rush', 'small_bottleneck', 'small_complex',
        'medium_normal', 'medium_stress', 'large_intro', 'large_advanced',
        'production_warmup', 'production_rush', 'production_heavy', 'production_expert'
    ]
    
    print("=" * 80)
    print("PHASE 3 TRAINING MONITOR")
    print("=" * 80)
    print()
    
    while True:
        state, logs = get_training_status()
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print(f"PHASE 3 TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # Overall progress
        completed = len(state.get('completed_stages', []))
        total = len(stages)
        progress = completed / total * 100
        
        print(f"Overall Progress: {completed}/{total} stages ({progress:.1f}%)")
        print()
        
        # Current stage
        current_idx = state.get('current_stage_idx', 0)
        if current_idx < len(stages):
            current_stage = stages[current_idx]
            print(f"Current Stage: {current_stage}")
        else:
            print("Training Complete!")
            break
            
        # Recent episodes
        print("\nRecent Episodes:")
        print("-" * 60)
        
        episodes = []
        for line in logs:
            episode = parse_episode_info(line)
            if episode:
                episodes.append(episode)
                
        # Show last 5 episodes
        for ep in episodes[-5:]:
            print(f"Episode {ep['episode']:3d} | "
                  f"Reward: {ep['reward']:6.1f} | "
                  f"Jobs: {ep['completed']:2d} | "
                  f"Late: {ep['late']:2d} | "
                  f"Util: {ep['utilization']:5.1f}%")
                  
        # Average performance
        if episodes:
            avg_reward = sum(e['reward'] for e in episodes[-10:]) / min(10, len(episodes))
            avg_util = sum(e['utilization'] for e in episodes[-10:]) / min(10, len(episodes))
            print(f"\nLast 10 Episodes Average:")
            print(f"  Reward: {avg_reward:.2f}")
            print(f"  Utilization: {avg_util:.1f}%")
            
        # Training speed
        print("\nTraining Speed:")
        for line in logs[-20:]:
            if 'fps' in line and '|' in line:
                print(line.strip())
                break
                
        # Completed stages
        print("\nCompleted Stages:")
        for i, stage in enumerate(state.get('completed_stages', [])):
            print(f"  ✓ {i+1}. {stage}")
            
        # Model files
        print("\nSaved Models:")
        checkpoint_dir = 'phase3/checkpoints'
        if os.path.exists(checkpoint_dir):
            models = [f for f in os.listdir(checkpoint_dir) if f.endswith('_final.zip')]
            for model in sorted(models):
                size = os.path.getsize(os.path.join(checkpoint_dir, model)) / 1024 / 1024
                print(f"  • {model} ({size:.1f} MB)")
                
        print("\nPress Ctrl+C to exit")
        print("-" * 80)
        
        # Wait before refresh
        time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")