"""Incremental improvement approach - start from best models and gradually improve"""

import os
import sys
import json
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class IncrementalWrapper(Monitor):
    """Wrapper for incremental improvement"""
    
    def __init__(self, env, target_rate=0.6):
        super().__init__(env)
        self.target_rate = target_rate
        self.episode_scheduled = 0
        
    def reset(self, **kwargs):
        self.episode_scheduled = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        
        # Reward based on current target
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            reward = 100.0
            self.episode_scheduled += 1
        elif info.get('action_valid', False):
            reward = 0.2
        else:
            reward = -1.0
        
        # Bonus for reaching target
        if done or truncated:
            total = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            rate = self.episode_scheduled / total if total > 0 else 0
            
            if rate >= self.target_rate:
                reward += 1000.0 * rate  # Proportional bonus
            else:
                reward += 100.0 * rate  # Still reward progress
        
        return obs, reward, done, truncated, info


def find_best_model(stage_name):
    """Find the best existing model for a stage"""
    model_paths = [
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/models_100_percent/{stage_name}_100.zip",
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/models_80_percent/{stage_name}_final.zip",
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/phased_models/{stage_name}_phased.zip",
        f"/Users/carrickcheah/Project/ppo/app_2/phase3/truly_fixed_models/{stage_name}_final.zip"
    ]
    
    best_model = None
    best_rate = 0
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = PPO.load(path)
                
                # Quick evaluation
                env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
                total_scheduled = 0
                total_possible = 0
                
                for _ in range(5):
                    obs, _ = env.reset()
                    done = False
                    steps = 0
                    
                    while not done and steps < 200:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, done, truncated, _ = env.step(action)
                        done = done or truncated
                        steps += 1
                    
                    scheduled = len(env.scheduled_jobs) if hasattr(env, 'scheduled_jobs') else 0
                    total = env.total_tasks if hasattr(env, 'total_tasks') else 1
                    total_scheduled += scheduled
                    total_possible += total
                
                rate = total_scheduled / total_possible if total_possible > 0 else 0
                
                if rate > best_rate:
                    best_rate = rate
                    best_model = model
                    print(f"Found model with {rate:.1%} performance: {path}")
            except:
                continue
    
    return best_model, best_rate


def incremental_train(stage_name, start_model=None, start_rate=0.0, target_increments=[0.6, 0.7, 0.8]):
    """Train incrementally towards 80%"""
    
    print(f"\nIncremental training for {stage_name}")
    print(f"Starting from: {start_rate:.1%}")
    print(f"Targets: {target_increments}")
    print("-" * 50)
    
    current_model = start_model
    current_rate = start_rate
    
    for target in target_increments:
        if current_rate >= target:
            print(f"Already at or above {target:.1%}, skipping...")
            continue
            
        print(f"\nTraining towards {target:.1%} target...")
        
        # Create environment with target
        base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        env = IncrementalWrapper(base_env, target_rate=target)
        env = DummyVecEnv([lambda: env])
        
        # Create or continue model
        if current_model is None:
            model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=2048, 
                       batch_size=128, ent_coef=0.1, verbose=0)
        else:
            model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=2048,
                       batch_size=128, ent_coef=0.1, verbose=0)
            # Copy policy weights
            model.policy.load_state_dict(current_model.policy.state_dict())
        
        # Train for shorter period
        model.learn(total_timesteps=100000)
        
        # Evaluate
        eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        total_scheduled = 0
        total_possible = 0
        
        for _ in range(10):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            
            while not done and steps < 300:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = eval_env.step(action)
                done = done or truncated
                steps += 1
            
            scheduled = len(eval_env.scheduled_jobs) if hasattr(eval_env, 'scheduled_jobs') else 0
            total = eval_env.total_tasks if hasattr(eval_env, 'total_tasks') else 1
            total_scheduled += scheduled
            total_possible += total
        
        new_rate = total_scheduled / total_possible if total_possible > 0 else 0
        print(f"Achieved: {new_rate:.1%} (target was {target:.1%})")
        
        if new_rate > current_rate:
            current_rate = new_rate
            current_model = model
            
            # Save if significant improvement
            if new_rate >= target - 0.05:  # Within 5% of target
                save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/incremental_models"
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"{stage_name}_{int(new_rate*100)}.zip")
                model.save(model_path)
                print(f"Saved model at {new_rate:.1%}")
        
        if current_rate >= 0.8:
            print(f"Reached 80% target!")
            break
    
    return current_rate


def main():
    stages_to_improve = ['toy_normal', 'toy_hard', 'toy_multi']
    
    print("INCREMENTAL IMPROVEMENT APPROACH")
    print("=" * 60)
    print("Starting from best existing models and improving gradually")
    
    results = {}
    
    for stage in stages_to_improve:
        print(f"\n\n{'='*60}")
        print(f"Stage: {stage}")
        print(f"{'='*60}")
        
        # Find best existing model
        best_model, best_rate = find_best_model(stage)
        
        # Train incrementally
        final_rate = incremental_train(stage, best_model, best_rate)
        results[stage] = final_rate
    
    # Summary
    print("\n\n" + "="*60)
    print("INCREMENTAL IMPROVEMENT RESULTS")
    print("="*60)
    
    for stage, rate in results.items():
        status = "✓" if rate >= 0.8 else "✗"
        print(f"{stage}: {rate:.1%} {status}")
    
    if all(r >= 0.8 for r in results.values()):
        print("\n✓ All stages reached 80% target!")
    else:
        print("\n✗ Some stages still below 80%")
        print("\nThe scheduling problem complexity and conflicting objectives")
        print("make it challenging to achieve high completion rates.")
        print("Consider using:")
        print("- Hierarchical RL to decompose the problem")
        print("- Imitation learning from the 100% sequences we found")
        print("- Different state representations or action spaces")


if __name__ == "__main__":
    main()