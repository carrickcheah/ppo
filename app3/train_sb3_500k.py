#!/usr/bin/env python
"""
Train SB3 PPO for 500k steps - 10x more than demo
This should achieve ~5x improvement over custom PPO
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environments.scheduling_env import SchedulingEnv
import torch
import numpy as np
import time
from datetime import datetime

def make_env(data_path: str):
    """Create environment for SB3."""
    def _init():
        env = SchedulingEnv(
            snapshot_path=data_path,
            max_steps=3000,
            planning_horizon=720.0,
            reward_config={
                'on_time_reward': 300.0,      # 3x bonus
                'early_bonus_per_day': 150.0,  # 3x bonus
                'late_penalty_per_day': -30.0, # Reduced penalty
                'utilization_bonus': 100.0,    # 10x bonus
                'action_taken_bonus': 30.0,    # 6x bonus
                'idle_penalty': -5.0,           # 5x penalty
                'sequence_violation_penalty': -50.0,  # Reduced
            }
        )
        return Monitor(env)
    return _init

def train_sb3_500k():
    """Train SB3 PPO for 500k steps with curriculum learning."""
    
    print("="*80)
    print("SB3 PPO TRAINING - 500K STEPS (5% OF FULL TRAINING)")
    print("="*80)
    print("\nTarget: 5x improvement over custom PPO")
    print("Expected efficiency: ~15% (vs 3% current)")
    print("-"*80)
    
    # Create directories
    os.makedirs("checkpoints/sb3_500k", exist_ok=True)
    os.makedirs("logs/sb3_500k", exist_ok=True)
    
    # Training stages with curriculum learning
    stages = [
        ('data/10_jobs.json', 100000, 'Stage 1: Simple'),
        ('data/20_jobs.json', 150000, 'Stage 2: Medium'),
        ('data/40_jobs.json', 250000, 'Stage 3: Complex'),
    ]
    
    total_timesteps = sum(steps for _, steps, _ in stages)
    print(f"\nTotal training steps: {total_timesteps:,}")
    print(f"Estimated time: ~10-15 minutes on M4 Pro")
    print("-"*80)
    
    model = None
    cumulative_steps = 0
    start_time = time.time()
    
    for stage_idx, (data_path, stage_steps, description) in enumerate(stages, 1):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {description}")
        print(f"Data: {data_path}")
        print(f"Steps: {stage_steps:,}")
        print(f"{'='*60}")
        
        # Create environment
        env = make_vec_env(
            make_env(data_path),
            n_envs=1,  # Single env for stability
            vec_env_cls=DummyVecEnv
        )
        
        # Create eval environment
        eval_env = make_vec_env(
            make_env(data_path),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
        
        if model is None:
            # Create new model for first stage
            print("\nCreating enhanced SB3 PPO model...")
            
            # Larger network for better capacity
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[1024, 512, 256, 128],  # Larger policy network
                    vf=[1024, 512, 256, 128]   # Separate value network
                ),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
            )
            
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=5e-4,  # Higher initial LR
                n_steps=2048,        # Larger rollout
                batch_size=128,
                n_epochs=15,         # More epochs
                gamma=0.995,         # Higher gamma for long-term planning
                gae_lambda=0.98,     # Higher GAE
                clip_range=0.2,
                ent_coef=0.02,       # More exploration
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.03,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"logs/sb3_500k/stage_{stage_idx}",
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f"Model architecture: {policy_kwargs['net_arch']}")
            print(f"Total parameters: ~4M")
        else:
            # Transfer to new environment
            print("\nTransferring model to new stage...")
            model.set_env(env)
            
            # Decay learning rate
            new_lr = 5e-4 * (0.7 ** (stage_idx - 1))
            model.learning_rate = new_lr
            print(f"Learning rate: {new_lr:.2e}")
        
        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"checkpoints/sb3_500k/stage_{stage_idx}",
            log_path=f"logs/sb3_500k/stage_{stage_idx}",
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path=f"checkpoints/sb3_500k",
            name_prefix=f"checkpoint_stage_{stage_idx}"
        )
        
        callback_list = CallbackList([eval_callback, checkpoint_callback])
        
        # Train
        print(f"\nTraining stage {stage_idx}...")
        stage_start = time.time()
        
        model.learn(
            total_timesteps=stage_steps,
            callback=callback_list,
            reset_num_timesteps=False,
            progress_bar=False
        )
        
        stage_time = time.time() - stage_start
        cumulative_steps += stage_steps
        
        print(f"\nStage {stage_idx} complete!")
        print(f"Stage time: {stage_time:.1f}s")
        print(f"Cumulative steps: {cumulative_steps:,}")
        print(f"Steps/second: {stage_steps/stage_time:.0f}")
        
        # Save stage model
        model.save(f"checkpoints/sb3_500k/model_stage_{stage_idx}")
        
        # Quick test
        print("\nQuick validation...")
        test_env = SchedulingEnv(data_path, max_steps=3000)
        obs, info = test_env.reset()
        done = False
        steps = 0
        
        while not done and steps < 3000:
            action, _ = model.predict(obs, deterministic=True)
            if 'action_mask' in info:
                mask = info['action_mask']
                if not mask[action]:
                    valid_actions = np.where(mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
            obs, reward, terminated, truncated, info = test_env.step(int(action))
            done = terminated or truncated
            steps += 1
        
        completion = info['tasks_scheduled'] / info['total_tasks'] * 100
        print(f"Completion: {completion:.1f}%")
        print(f"Reward: {test_env.episode_reward:.0f}")
        
        # Cleanup
        env.close()
        eval_env.close()
    
    # Save final model
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    model.save("checkpoints/sb3_500k/final_500k_model")
    
    print(f"\n📊 Training Statistics:")
    print(f"- Total steps: {cumulative_steps:,}")
    print(f"- Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"- Average speed: {cumulative_steps/total_time:.0f} steps/second")
    print(f"- Model saved: checkpoints/sb3_500k/final_500k_model.zip")
    
    # Final validation on 100 jobs
    print(f"\n{'='*60}")
    print("FINAL VALIDATION ON 100 JOBS")
    print(f"{'='*60}")
    
    if os.path.exists('data/100_jobs.json'):
        test_env = SchedulingEnv('data/100_jobs.json', max_steps=10000)
        obs, info = test_env.reset()
        
        start_time = time.time()
        done = False
        steps = 0
        
        while not done and steps < 10000:
            action, _ = model.predict(obs, deterministic=True)
            if 'action_mask' in info:
                mask = info['action_mask']
                if not mask[action]:
                    valid_actions = np.where(mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
            obs, reward, terminated, truncated, info = test_env.step(int(action))
            done = terminated or truncated
            steps += 1
        
        test_time = time.time() - start_time
        
        # Calculate efficiency
        schedule = test_env.get_final_schedule()
        if schedule['tasks']:
            total_processing = sum(t['processing_time'] for t in schedule['tasks'])
            makespan = max(t['end'] for t in schedule['tasks'])
            n_machines = len(test_env.loader.machines)
            efficiency = (total_processing / n_machines / makespan * 100)
        else:
            efficiency = 0
        
        print(f"\nResults on 100 jobs:")
        print(f"- Completion: {info['tasks_scheduled']}/{info['total_tasks']} ({info['tasks_scheduled']/info['total_tasks']*100:.1f}%)")
        print(f"- Efficiency: {efficiency:.1f}%")
        print(f"- Reward: {test_env.episode_reward:.0f}")
        print(f"- Time: {test_time:.2f}s")
        print(f"- Steps: {steps}")
        
        # Compare with custom PPO baseline (7.4% efficiency)
        improvement = efficiency / 7.4
        print(f"\n🎯 IMPROVEMENT vs CUSTOM PPO:")
        print(f"- Custom PPO efficiency: 7.4%")
        print(f"- SB3 500k efficiency: {efficiency:.1f}%")
        print(f"- Improvement factor: {improvement:.1f}x")
        
        if improvement >= 5:
            print("\n✅ TARGET ACHIEVED! 5x improvement reached!")
        else:
            print(f"\n📈 Progress: {improvement:.1f}x of 5x target")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Run: python validate_500k_model.py")
    print("2. Schedule jobs: python schedule_with_500k.py")
    print("3. Continue to 1M steps: python train_sb3_1m.py")
    print(f"{'='*80}")
    
    return model

if __name__ == "__main__":
    model = train_sb3_500k()