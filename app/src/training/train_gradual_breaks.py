"""
Gradual break introduction for Phase 2 curriculum learning.
Phase 2a: Only tea breaks (30 min/day)
Phase 2b: Add lunch (1.5 hours/day)
Phase 2c: Add dinner and weekends (full breaks)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import time
from datetime import datetime
import json
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.environments.scaled_production_env import ScaledProductionEnv


class GradualBreaksEnv(ScaledProductionEnv):
    """Environment with configurable break levels."""
    
    def __init__(self, break_level='none', **kwargs):
        """
        Initialize with specific break level.
        
        Args:
            break_level: 'none', 'tea', 'tea_lunch', or 'full'
        """
        self.break_level = break_level
        
        # Initialize with breaks disabled first
        super().__init__(use_break_constraints=False, **kwargs)
        
        # Then configure breaks based on level
        if break_level != 'none':
            self.use_break_constraints = True
            self._configure_breaks()
    
    def _configure_breaks(self):
        """Configure breaks based on level."""
        if self.break_level == 'tea':
            # Only morning and afternoon tea (30 min total)
            self.break_times = [
                {'start': 10.0, 'end': 10.25, 'name': 'Morning Tea'},
                {'start': 15.0, 'end': 15.25, 'name': 'Afternoon Tea'}
            ]
        elif self.break_level == 'tea_lunch':
            # Tea breaks + lunch (2 hours total)
            self.break_times = [
                {'start': 10.0, 'end': 10.25, 'name': 'Morning Tea'},
                {'start': 12.0, 'end': 13.0, 'name': 'Lunch'},
                {'start': 15.0, 'end': 15.25, 'name': 'Afternoon Tea'}
            ]
        elif self.break_level == 'full':
            # All breaks including dinner and weekends
            self.break_times = [
                {'start': 10.0, 'end': 10.25, 'name': 'Morning Tea'},
                {'start': 12.0, 'end': 13.0, 'name': 'Lunch'},
                {'start': 15.0, 'end': 15.25, 'name': 'Afternoon Tea'},
                {'start': 18.0, 'end': 19.0, 'name': 'Dinner'}
            ]
            # Weekends handled by parent class
        else:
            self.break_times = []


def create_env_phase2a(seed=None):
    """Phase 2a: Only tea breaks."""
    env = GradualBreaksEnv(
        break_level='tea',
        n_machines=40,
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=seed
    )
    return Monitor(env)


def create_env_phase2b(seed=None):
    """Phase 2b: Tea + lunch breaks."""
    env = GradualBreaksEnv(
        break_level='tea_lunch',
        n_machines=40,
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=seed
    )
    return Monitor(env)


def create_env_phase2c(seed=None):
    """Phase 2c: Full breaks (tea + lunch + dinner + weekends)."""
    env = GradualBreaksEnv(
        break_level='full',
        n_machines=40,
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=seed
    )
    return Monitor(env)


def train_gradual_phase(phase_name, create_env_fn, pretrained_model_path, 
                       config, save_path, target_makespan=None):
    """Train a gradual break phase."""
    print(f"\n{'='*60}")
    print(f"TRAINING {phase_name}")
    print(f"{'='*60}")
    
    # Create environments
    env = make_vec_env(
        create_env_fn,
        n_envs=config['training']['n_envs'],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'seed': config['training']['seed']}
    )
    
    # Load pretrained model
    print(f"Loading model from: {pretrained_model_path}")
    model = PPO.load(pretrained_model_path, env=env)
    
    # Adjust learning rate for fine-tuning
    model.learning_rate = config['training']['learning_rate'] * 0.5
    print(f"Learning rate: {model.learning_rate}")
    
    # Training callback
    best_makespan = float('inf')
    episode_count = 0
    start_time = time.time()
    
    def callback(locals_, globals_):
        nonlocal episode_count, best_makespan
        
        if "infos" in locals_:
            for info in locals_["infos"]:
                if "episode" in info:
                    episode_count += 1
                    
                    if episode_count % 50 == 0:
                        # Quick evaluation
                        eval_env = create_env_fn(seed=123)
                        obs, _ = eval_env.reset()
                        done = False
                        
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, _ = eval_env.step(action)
                            done = terminated or truncated
                        
                        base_env = eval_env.unwrapped if hasattr(eval_env, 'unwrapped') else eval_env
                        makespan = base_env.episode_makespan
                        
                        elapsed = (time.time() - start_time) / 60
                        print(f"\n[Episode {episode_count}] Time: {elapsed:.1f}min")
                        print(f"Current makespan: {makespan:.2f}h")
                        
                        if makespan < best_makespan:
                            best_makespan = makespan
                            print(f"★ New best: {best_makespan:.2f}h")
                            
                            # Save if meeting target
                            if target_makespan and makespan <= target_makespan:
                                print(f"✓ Target achieved! Saving model...")
                                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                                model.save(save_path.replace('.zip', '_best'))
        
        return True
    
    # Train
    print(f"\nTraining for {config['training']['gradual_timesteps']:,} timesteps...")
    if target_makespan:
        print(f"Target makespan: {target_makespan:.1f}h")
    
    model.learn(
        total_timesteps=config['training']['gradual_timesteps'],
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    # Save final model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    
    # Final evaluation
    print(f"\nEvaluating {phase_name} performance...")
    makespans = []
    
    for _ in range(5):
        eval_env = create_env_fn(seed=np.random.randint(1000))
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        
        base_env = eval_env.unwrapped if hasattr(eval_env, 'unwrapped') else eval_env
        makespans.append(base_env.episode_makespan)
    
    avg_makespan = np.mean(makespans)
    training_time = (time.time() - start_time) / 60
    
    results = {
        'phase': phase_name,
        'avg_makespan': float(avg_makespan),
        'best_makespan': float(best_makespan),
        'training_time_min': float(training_time),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{phase_name} Results:")
    print(f"Average makespan: {avg_makespan:.2f}h")
    print(f"Best makespan: {best_makespan:.2f}h")
    print(f"Training time: {training_time:.1f} minutes")
    
    return results, avg_makespan


def main():
    print("\n" + "="*60)
    print("GRADUAL BREAK INTRODUCTION FOR PHASE 2")
    print("="*60)
    print("Strategy: Gradually add breaks to improve on 19.7h baseline")
    
    # Load config
    config_path = Path("configs/scaled_production_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add gradual training config
    config['training']['gradual_timesteps'] = 300000  # Shorter training per phase
    
    # Phase 1 baseline
    phase1_makespan = 16.2  # From previous training
    current_best = 19.7     # Phase 2 with full breaks
    
    results = {}
    
    # Phase 2a: Tea breaks only
    print("\n" + "-"*60)
    print("PHASE 2a: Adding tea breaks only (30 min/day)")
    print("Expected impact: ~2-3% increase from Phase 1")
    print("-"*60)
    
    phase1_model = "models/curriculum/phase1_no_breaks/final_model"
    
    results['phase2a'], makespan_2a = train_gradual_phase(
        "Phase 2a: Tea Breaks",
        create_env_phase2a,
        phase1_model,
        config,
        "models/curriculum/phase2a_tea_breaks/final_model",
        target_makespan=17.5  # Target: Phase 1 + 8%
    )
    
    # Phase 2b: Add lunch
    if makespan_2a < current_best:
        print("\n" + "-"*60)
        print("PHASE 2b: Adding lunch break (2 hours/day total)")
        print("Building on Phase 2a success")
        print("-"*60)
        
        results['phase2b'], makespan_2b = train_gradual_phase(
            "Phase 2b: Tea + Lunch",
            create_env_phase2b,
            "models/curriculum/phase2a_tea_breaks/final_model",
            config,
            "models/curriculum/phase2b_tea_lunch/final_model",
            target_makespan=18.5  # Target: Phase 1 + 14%
        )
        
        # Phase 2c: Full breaks
        if makespan_2b < current_best:
            print("\n" + "-"*60)
            print("PHASE 2c: Adding dinner + weekends (full breaks)")
            print("Final phase of gradual introduction")
            print("-"*60)
            
            results['phase2c'], makespan_2c = train_gradual_phase(
                "Phase 2c: Full Breaks",
                create_env_phase2c,
                "models/curriculum/phase2b_tea_lunch/final_model",
                config,
                "models/curriculum/phase2c_full_breaks/final_model",
                target_makespan=19.4  # Target: Beat baseline!
            )
    
    # Save all results
    results_path = "logs/curriculum/gradual_breaks_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("GRADUAL BREAK INTRODUCTION SUMMARY")
    print("="*60)
    
    print(f"Phase 1 (no breaks): {phase1_makespan:.1f}h")
    
    for phase_key in ['phase2a', 'phase2b', 'phase2c']:
        if phase_key in results:
            phase = results[phase_key]
            print(f"{phase['phase']}: {phase['avg_makespan']:.2f}h")
    
    print(f"\nOriginal Phase 2 (direct to full breaks): {current_best:.1f}h")
    
    # Check if we beat the baseline
    final_makespan = results.get('phase2c', {}).get('avg_makespan', 
                                results.get('phase2b', {}).get('avg_makespan',
                                results.get('phase2a', {}).get('avg_makespan', 99)))
    
    if final_makespan < 19.4:
        print(f"\n✓ SUCCESS! Beat baseline: {final_makespan:.2f}h < 19.4h")
    else:
        print(f"\n⚠ Did not beat baseline: {final_makespan:.2f}h > 19.4h")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()