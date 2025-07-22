#!/usr/bin/env python3
"""
Extended Phase 5 training - 2M timesteps for full performance
Target: <45 hour makespan (from Phase 4's 49.2 hours)
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import time
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)

from src.environments.multidiscrete_hierarchical_env import MultiDiscreteHierarchicalEnv

class MakespanCallback(BaseCallback):
    """Custom callback to track makespan improvement"""
    
    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_makespan = float('inf')
        self.makespans = []
        self.target_makespan = 45.0  # Target from Phase 4
        
    def _on_step(self) -> bool:
        # Every 50k steps, evaluate makespan
        if self.n_calls % 50000 == 0:
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.eval_env.step(action)
            
            if info[0].get('scheduled_count', 0) == 411:
                makespan = info[0].get('makespan', float('inf'))
                self.makespans.append((self.n_calls, makespan))
                
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    print(f"\n🎯 New best makespan: {makespan:.1f} hours (target: <{self.target_makespan}h)")
                    
                    if makespan < self.target_makespan:
                        print(f"\n🎉 TARGET ACHIEVED! Makespan {makespan:.1f}h < {self.target_makespan}h")
                        improvement = (49.2 - makespan) / 49.2 * 100
                        print(f"Improvement: {improvement:.1f}% from Phase 4")
        
        return True

def create_env(rank: int = 0, eval_env: bool = False):
    """Create environment for training"""
    def _init():
        env = MultiDiscreteHierarchicalEnv(
            n_machines=145,
            n_jobs=411,
            snapshot_file="data/real_production_snapshot.json",
            max_episode_steps=1000,
            use_break_constraints=True,
            use_holiday_constraints=True,
            invalid_action_penalty=-10.0,
            seed=42 + rank if not eval_env else 123
        )
        return Monitor(env)
    return _init

def train_extended():
    print("\n" + "="*60)
    print("Phase 5 Extended Training - Target <45h Makespan")
    print("="*60 + "\n")
    
    # Configuration
    config = {
        'total_timesteps': 2000000,  # 2M steps
        'n_envs': 8,  # Parallel environments
        'learning_rate': 0.0003,
        'batch_size': 256,
        'n_epochs': 10,
        'checkpoint_freq': 100000,
        'eval_freq': 50000,
        'load_checkpoint': "models/multidiscrete/fixed/phase5_fixed_100000_steps.zip"  # Start from 100k
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create environments
    print(f"\nCreating {config['n_envs']} training environments...")
    if config['n_envs'] > 1:
        train_envs = SubprocVecEnv([create_env(i) for i in range(config['n_envs'])])
    else:
        train_envs = DummyVecEnv([create_env(0)])
    
    # Create eval environment
    eval_env = DummyVecEnv([create_env(eval_env=True)])
    
    # Create or load model
    if config['load_checkpoint'] and os.path.exists(config['load_checkpoint']):
        print(f"\nLoading checkpoint: {config['load_checkpoint']}")
        model = PPO.load(config['load_checkpoint'], env=train_envs)
        print("Checkpoint loaded - continuing from 100k steps")
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_envs,
            learning_rate=config['learning_rate'],
            n_steps=2048 // config['n_envs'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'] // config['n_envs'],
        save_path="models/multidiscrete/extended",
        name_prefix="phase5_extended"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/multidiscrete/extended/best",
        log_path="logs/phase5/extended",
        eval_freq=config['eval_freq'] // config['n_envs'],
        n_eval_episodes=3,
        deterministic=True
    )
    
    makespan_callback = MakespanCallback(eval_env)
    
    callbacks = CallbackList([checkpoint_callback, eval_callback, makespan_callback])
    
    # Train
    print(f"\nStarting extended training for {config['total_timesteps']:,} timesteps...")
    print("Expected time: 30-40 minutes\n")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config['total_timesteps'] - 100000,  # Subtract already trained steps
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False  # Continue from checkpoint
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time/60:.1f} minutes")
    
    # Save final model
    model.save("models/multidiscrete/extended/final_extended_model")
    print("Final model saved")
    
    # Final evaluation
    print("\n" + "="*40)
    print("Final Evaluation")
    print("="*40)
    
    obs = eval_env.reset()
    done = False
    steps = 0
    scheduled = 0
    invalid = 0
    
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        steps += 1
        
        if info[0].get('invalid_action', False):
            invalid += 1
        
        scheduled = info[0].get('scheduled_count', 0)
        
        if steps % 100 == 0:
            print(f"  Step {steps}: {scheduled} jobs scheduled")
    
    print(f"\nFinal Results:")
    print(f"  Jobs scheduled: {scheduled}/411")
    print(f"  Invalid action rate: {invalid/steps*100:.1f}%")
    
    if scheduled == 411:
        makespan = info[0].get('makespan', 0)
        print(f"  Makespan: {makespan:.1f} hours")
        print(f"\n📊 Phase 5 vs Phase 4 Comparison:")
        print(f"  Phase 4 (batch): 49.2 hours")
        print(f"  Phase 5 (hier): {makespan:.1f} hours")
        improvement = (49.2 - makespan) / 49.2 * 100
        print(f"  Improvement: {improvement:.1f}%")
        
        if makespan < 45:
            print(f"\n✅ SUCCESS! Target achieved: {makespan:.1f}h < 45h")
        else:
            print(f"\n⚠️  Close but not quite: {makespan:.1f}h (target: <45h)")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_timesteps': config['total_timesteps'],
        'final_scheduled': scheduled,
        'final_makespan': makespan if scheduled == 411 else None,
        'invalid_rate': invalid/steps*100,
        'training_time_minutes': train_time/60,
        'best_makespans': makespan_callback.makespans
    }
    
    os.makedirs("results/phase5", exist_ok=True)
    with open("results/phase5/extended_training_results.yaml", 'w') as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to: results/phase5/extended_training_results.yaml")

if __name__ == "__main__":
    train_extended()