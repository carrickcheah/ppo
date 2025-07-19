"""
Phase 3: Add holidays to the curriculum learning approach.
Builds on Phase 2 (with breaks) to add holiday constraints.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import time
from datetime import datetime, timedelta
import json
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.environments.scaled_production_env import ScaledProductionEnv


class Phase3HolidayEnv(ScaledProductionEnv):
    """Environment with breaks and holidays."""
    
    def __init__(self, holidays=None, **kwargs):
        """
        Initialize with holiday constraints.
        
        Args:
            holidays: List of holiday dictionaries with 'start' and 'end' times
        """
        # Initialize with breaks enabled
        super().__init__(use_break_constraints=True, **kwargs)
        
        # Add holiday constraints
        self.holidays = holidays or self._get_default_holidays()
        
    def _get_default_holidays(self):
        """Get default holidays for testing."""
        # Example holidays (in hours from start)
        holidays = [
            # Weekend (Saturday noon to Monday 6am - already in breaks)
            # Public holidays (full days)
            {'start': 24 * 7, 'end': 24 * 8, 'name': 'Public Holiday 1'},  # Week 2
            {'start': 24 * 14, 'end': 24 * 15, 'name': 'Public Holiday 2'},  # Week 3
            {'start': 24 * 21, 'end': 24 * 22, 'name': 'Public Holiday 3'},  # Week 4
            # Half-day holidays
            {'start': 24 * 10 + 12, 'end': 24 * 11, 'name': 'Half Day Holiday'},
        ]
        return holidays
    
    def _is_holiday_time(self, start_time, end_time):
        """Check if time period overlaps with holidays."""
        for holiday in self.holidays:
            # Check overlap
            if start_time < holiday['end'] and end_time > holiday['start']:
                return True
        return False
    
    def _find_valid_start_time(self, machine_idx, start_time, processing_time, max_attempts=100):
        """Find valid start time avoiding breaks and holidays."""
        current_time = start_time
        
        for _ in range(max_attempts):
            # Check breaks (from parent class)
            if self.use_break_constraints:
                # Check if time overlaps with breaks
                valid = True
                
                # Check daily breaks
                for break_time in getattr(self, 'break_times', []):
                    if current_time < break_time['end'] and current_time + processing_time > break_time['start']:
                        valid = False
                        current_time = break_time['end']
                        break
                
                # Check weekends
                if valid:
                    # Simple weekend check (Saturday noon to Monday 6am)
                    day_of_week = int(current_time / 24) % 7
                    hour_of_day = current_time % 24
                    
                    if (day_of_week == 5 and hour_of_day >= 12) or \
                       (day_of_week == 6) or \
                       (day_of_week == 0 and hour_of_day < 6):
                        # Skip to Monday 6am
                        days_to_monday = (7 - day_of_week) % 7
                        if days_to_monday == 0 and hour_of_day >= 6:
                            days_to_monday = 7
                        current_time = int(current_time / 24) * 24 + days_to_monday * 24 + 6
                        valid = False
                
                if not valid:
                    continue
            
            # Check holidays
            if self._is_holiday_time(current_time, current_time + processing_time):
                # Find next available time after holiday
                for holiday in self.holidays:
                    if current_time < holiday['end'] and current_time + processing_time > holiday['start']:
                        current_time = holiday['end']
                        break
                continue
            
            # If we get here, time is valid
            return current_time
        
        # If no valid time found, return original (will likely fail)
        return start_time
    
    def reset(self, **kwargs):
        """Reset and log holiday info."""
        obs, info = super().reset(**kwargs)
        
        # Add holiday info
        total_holiday_hours = sum(h['end'] - h['start'] for h in self.holidays)
        info['holiday_hours'] = total_holiday_hours
        info['num_holidays'] = len(self.holidays)
        
        return obs, info


def create_env_phase3(seed=None):
    """Create Phase 3 environment with breaks and holidays."""
    env = Phase3HolidayEnv(
        n_machines=40,
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=seed
    )
    return Monitor(env)


def train_phase3(config):
    """Train Phase 3 with holidays."""
    print("\n" + "="*60)
    print("PHASE 3: ADDING HOLIDAYS")
    print("="*60)
    print("Building on Phase 2 (breaks) to add holiday constraints")
    
    # Create environments
    env = make_vec_env(
        create_env_phase3,
        n_envs=config['training']['n_envs'],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'seed': config['training']['seed']}
    )
    
    # Load Phase 2 model
    phase2_model_path = "models/curriculum/phase2_with_breaks/final_model"
    
    if not Path(phase2_model_path + ".zip").exists():
        print("ERROR: Phase 2 model not found!")
        print("Please run Phase 2 training first.")
        return None
    
    print(f"\nLoading Phase 2 model from: {phase2_model_path}")
    model = PPO.load(phase2_model_path, env=env)
    
    # Adjust for fine-tuning
    model.learning_rate = config['training']['learning_rate'] * 0.3
    model.ent_coef = 0.005
    
    print(f"Learning rate: {model.learning_rate}")
    print(f"Entropy coefficient: {model.ent_coef}")
    
    # Training tracking
    best_makespan = float('inf')
    episode_count = 0
    episode_makespans = []
    start_time = time.time()
    
    def callback(locals_, globals_):
        nonlocal episode_count, best_makespan, episode_makespans
        
        if "infos" in locals_:
            for info in locals_["infos"]:
                if "episode" in info:
                    episode_count += 1
                    
                    # Track makespan if available
                    if "makespan" in info:
                        episode_makespans.append(info["makespan"])
                    
                    if episode_count % 100 == 0:
                        avg_makespan = np.mean(episode_makespans[-50:]) if episode_makespans else 0
                        elapsed = (time.time() - start_time) / 60
                        
                        print(f"\n[Episode {episode_count}] Time: {elapsed:.1f}min")
                        print(f"Average makespan (last 50): {avg_makespan:.2f}h")
                        
                        # Quick evaluation
                        eval_env = create_env_phase3(seed=123)
                        obs, _ = eval_env.reset()
                        done = False
                        
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, _ = eval_env.step(action)
                            done = terminated or truncated
                        
                        base_env = eval_env.unwrapped if hasattr(eval_env, 'unwrapped') else eval_env
                        current_makespan = base_env.episode_makespan
                        
                        print(f"Evaluation makespan: {current_makespan:.2f}h")
                        
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            print(f"★ New best: {best_makespan:.2f}h")
                            
                            # Save best model
                            Path("models/curriculum/phase3_holidays/best").mkdir(parents=True, exist_ok=True)
                            model.save("models/curriculum/phase3_holidays/best/best_model")
        
        return True
    
    # Train
    print(f"\nTraining for {config['training']['phase3_timesteps']:,} timesteps...")
    print("Target: Maintain performance < 20h with holidays")
    print("-" * 60)
    
    model.learn(
        total_timesteps=config['training']['phase3_timesteps'],
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    # Save final model
    save_path = "models/curriculum/phase3_holidays/final_model"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    
    # Final evaluation
    print("\n" + "="*60)
    print("PHASE 3 EVALUATION")
    print("="*60)
    
    makespans = []
    holiday_impacts = []
    
    for i in range(10):
        eval_env = create_env_phase3(seed=i*100)
        obs, info = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        
        base_env = eval_env.unwrapped if hasattr(eval_env, 'unwrapped') else eval_env
        makespan = base_env.episode_makespan
        makespans.append(makespan)
        
        # Calculate holiday impact
        holiday_hours = info.get('holiday_hours', 0)
        holiday_impacts.append(holiday_hours)
    
    avg_makespan = np.mean(makespans)
    training_time = (time.time() - start_time) / 60
    
    results = {
        'phase': 'Phase 3: Holidays',
        'avg_makespan': float(avg_makespan),
        'best_makespan': float(best_makespan),
        'training_time_min': float(training_time),
        'holiday_hours': float(np.mean(holiday_impacts)),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nFinal Results:")
    print(f"Average makespan: {avg_makespan:.2f}h")
    print(f"Best makespan: {best_makespan:.2f}h")
    print(f"Holiday hours per schedule: {np.mean(holiday_impacts):.1f}h")
    print(f"Training time: {training_time:.1f} minutes")
    
    return results


def main():
    print("\n" + "="*60)
    print("CURRICULUM LEARNING - PHASE 3: HOLIDAYS")
    print("="*60)
    
    # Load config
    config_path = Path("configs/scaled_production_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add Phase 3 config
    config['training']['phase3_timesteps'] = 500000
    
    # Previous results for comparison
    phase1_makespan = 16.2  # No breaks
    phase2_makespan = 19.7  # With breaks
    baseline = 19.4         # Random policy
    
    print(f"\nPrevious Results:")
    print(f"Phase 1 (no breaks): {phase1_makespan:.1f}h")
    print(f"Phase 2 (with breaks): {phase2_makespan:.1f}h")
    print(f"Baseline: {baseline:.1f}h")
    
    # Train Phase 3
    results = train_phase3(config)
    
    if results:
        # Save results
        results_path = "logs/curriculum/phase3_results.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        print("\n" + "="*60)
        print("CURRICULUM LEARNING COMPLETE")
        print("="*60)
        
        print(f"\nProgression:")
        print(f"Phase 1 (no constraints): {phase1_makespan:.1f}h")
        print(f"Phase 2 (+ breaks): {phase2_makespan:.1f}h ({(phase2_makespan/phase1_makespan-1)*100:.1f}% increase)")
        print(f"Phase 3 (+ holidays): {results['avg_makespan']:.1f}h ({(results['avg_makespan']/phase2_makespan-1)*100:.1f}% increase)")
        
        print(f"\nComparison to baseline ({baseline:.1f}h):")
        if results['avg_makespan'] < baseline:
            print(f"✓ STILL BEATING BASELINE by {baseline - results['avg_makespan']:.1f}h!")
        elif results['avg_makespan'] < 20.0:
            print(f"⚠ Above baseline by {results['avg_makespan'] - baseline:.1f}h but under 20h target")
        else:
            print(f"✗ Exceeds 20h target: {results['avg_makespan']:.1f}h")
        
        print(f"\nModels saved:")
        print(f"- Final: models/curriculum/phase3_holidays/final_model")
        print(f"- Best: models/curriculum/phase3_holidays/best/best_model")
        print(f"- Results: {results_path}")
    else:
        print("\nPhase 3 training failed. Check Phase 2 model exists.")


if __name__ == "__main__":
    main()