"""
Fixed gradual break introduction training with proper constraint enforcement.
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from src.environments.gradual_breaks_env import GradualBreaksEnv


def create_gradual_env(break_level='none'):
    """Create environment with specified break level."""
    def _init():
        env = GradualBreaksEnv(
            break_level=break_level,
            n_machines=40,
            data_file='data/large_production_data.json',
            snapshot_file='data/production_snapshot_latest.json',
            seed=None
        )
        return Monitor(env)
    return _init


def evaluate_performance(model, break_level, n_episodes=5):
    """Evaluate model performance on specific break level."""
    env = create_gradual_env(break_level)()
    
    makespans = []
    break_hours = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        if hasattr(base_env, 'episode_makespan'):
            makespans.append(base_env.episode_makespan)
        break_hours.append(info.get('break_hours_per_week', 0))
    
    return {
        'makespan': np.mean(makespans) if makespans else 0,
        'std': np.std(makespans) if makespans else 0,
        'break_hours': np.mean(break_hours)
    }


def train_gradual_phase(phase_name, break_level, pretrained_model_path, 
                       config, save_path, target_makespan=None):
    """Train a gradual break phase with proper enforcement."""
    print(f"\n{'='*60}")
    print(f"TRAINING {phase_name}")
    print(f"Break Level: {break_level}")
    print(f"{'='*60}")
    
    # Create environments
    env = make_vec_env(
        create_gradual_env(break_level),
        n_envs=config['training']['n_envs'],
        vec_env_cls=SubprocVecEnv
    )
    
    # Create eval environment
    eval_env = create_gradual_env(break_level)()
    
    # Load pretrained model
    if Path(pretrained_model_path + ".zip").exists():
        print(f"Loading model from: {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=env)
        model.learning_rate = config['training']['learning_rate'] * 0.5
    else:
        print("WARNING: Pretrained model not found, starting fresh")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config['training']['learning_rate'],
            n_steps=config['training']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['training']['n_epochs'],
            gamma=config['training']['gamma'],
            verbose=0
        )
    
    print(f"Learning rate: {model.learning_rate}")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/gradual/{break_level}/best/",
        log_path=f"logs/gradual/{break_level}/",
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"models/gradual/{break_level}/checkpoints/",
        name_prefix=f"{break_level}_checkpoint"
    )
    
    # Custom tracking
    best_makespan = float('inf')
    start_time = time.time()
    
    class ProgressCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.best_makespan = float('inf')
            
        def _on_step(self) -> bool:
            if self.n_calls % 10000 == 0:
                # Quick evaluation
                test_env = create_gradual_env(break_level)()
                obs, _ = test_env.reset()
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = test_env.step(action)
                    done = terminated or truncated
                
                base_env = test_env.unwrapped
                if hasattr(base_env, 'episode_makespan'):
                    makespan = base_env.episode_makespan
                    
                    if makespan < self.best_makespan:
                        self.best_makespan = makespan
                        print(f"\n[Step {self.n_calls}] New best makespan: {self.best_makespan:.2f}h")
                        
                        if target_makespan and makespan <= target_makespan:
                            print(f"✓ Target achieved! ({makespan:.2f}h <= {target_makespan:.1f}h)")
            
            return True
    
    progress_callback = ProgressCallback()
    
    # Train
    print(f"\nTraining for {config['training']['gradual_timesteps']:,} timesteps...")
    if target_makespan:
        print(f"Target makespan: {target_makespan:.1f}h")
    
    model.learn(
        total_timesteps=config['training']['gradual_timesteps'],
        callback=[eval_callback, checkpoint_callback, progress_callback],
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    # Save final model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    
    # Final evaluation
    print(f"\nEvaluating {phase_name} performance...")
    results = evaluate_performance(model, break_level, n_episodes=10)
    
    training_time = (time.time() - start_time) / 60
    
    final_results = {
        'phase': phase_name,
        'break_level': break_level,
        'avg_makespan': float(results['makespan']),
        'std_makespan': float(results['std']),
        'best_makespan': float(progress_callback.best_makespan),
        'break_hours': float(results['break_hours']),
        'training_time_min': float(training_time),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{phase_name} Results:")
    print(f"Average makespan: {results['makespan']:.2f}h (±{results['std']:.2f})")
    print(f"Best makespan: {progress_callback.best_makespan:.2f}h")
    print(f"Break hours/week: {results['break_hours']:.1f}h")
    print(f"Training time: {training_time:.1f} minutes")
    
    return final_results, results['makespan']


def main():
    print("\n" + "="*60)
    print("FIXED GRADUAL BREAK INTRODUCTION")
    print("="*60)
    print("Strategy: Gradually add breaks to beat 19.4h baseline")
    
    # Load config
    config_path = Path("configs/scaled_production_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for gradual training
    config['training']['gradual_timesteps'] = 400000
    config['training']['n_envs'] = 8
    
    # Track results
    all_results = {}
    phase1_model = "models/curriculum/phase1_no_breaks/final_model"
    
    # Phase 2a: Tea breaks only
    print("\n" + "-"*60)
    print("PHASE 2a: Tea breaks only (30 min/day)")
    print("Expected: ~17.0h (5% increase from 16.2h)")
    print("-"*60)
    
    results_2a, makespan_2a = train_gradual_phase(
        "Phase 2a: Tea Breaks",
        "tea",
        phase1_model,
        config,
        "models/gradual/tea/final_model",
        target_makespan=17.2
    )
    all_results['phase2a'] = results_2a
    
    # Phase 2b: Tea + Lunch
    if makespan_2a < 19.0:
        print("\n" + "-"*60)
        print("PHASE 2b: Tea + Lunch (2 hours/day)")
        print("Building on Phase 2a success")
        print("-"*60)
        
        results_2b, makespan_2b = train_gradual_phase(
            "Phase 2b: Tea + Lunch",
            "tea_lunch",
            "models/gradual/tea/final_model",
            config,
            "models/gradual/tea_lunch/final_model",
            target_makespan=18.5
        )
        all_results['phase2b'] = results_2b
        
        # Phase 2c: Full breaks
        if makespan_2b < 19.4:
            print("\n" + "-"*60)
            print("PHASE 2c: Full breaks (all daily + weekends)")
            print("Final push to beat baseline")
            print("-"*60)
            
            results_2c, makespan_2c = train_gradual_phase(
                "Phase 2c: Full Breaks",
                "full",
                "models/gradual/tea_lunch/final_model",
                config,
                "models/gradual/full/final_model",
                target_makespan=19.3  # Beat baseline!
            )
            all_results['phase2c'] = results_2c
    
    # Save results
    results_path = "logs/gradual/gradual_fixed_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("GRADUAL BREAK TRAINING SUMMARY")
    print("="*60)
    
    print(f"\nProgression:")
    print(f"Phase 1 (no breaks): 16.2h")
    
    for phase in ['phase2a', 'phase2b', 'phase2c']:
        if phase in all_results:
            r = all_results[phase]
            print(f"{r['phase']}: {r['avg_makespan']:.2f}h (±{r['std_makespan']:.2f})")
            print(f"  Break hours: {r['break_hours']:.1f}h/week")
            print(f"  Best achieved: {r['best_makespan']:.2f}h")
    
    print(f"\nBaseline to beat: 19.4h")
    
    # Check final result
    final_phase = 'phase2c' if 'phase2c' in all_results else \
                  'phase2b' if 'phase2b' in all_results else 'phase2a'
    
    if final_phase in all_results:
        final_makespan = all_results[final_phase]['avg_makespan']
        
        if final_makespan < 19.4:
            improvement = (19.4 - final_makespan) / 19.4 * 100
            print(f"\n✓ SUCCESS! Beat baseline by {improvement:.1f}%")
            print(f"Final makespan: {final_makespan:.2f}h < 19.4h")
        else:
            gap = final_makespan - 19.4
            print(f"\n⚠ Missed baseline by {gap:.2f}h")
            print(f"Final makespan: {final_makespan:.2f}h > 19.4h")
    
    print(f"\nResults saved to: {results_path}")
    print("\nModels saved:")
    print("- models/gradual/tea/final_model")
    print("- models/gradual/tea_lunch/final_model")
    print("- models/gradual/full/final_model")


if __name__ == "__main__":
    main()