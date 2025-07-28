"""Final push to achieve 100% on all toy stages"""

import os
import sys
import json
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class MaxCompletionWrapper(Monitor):
    """Maximize completion at all costs"""
    
    def __init__(self, env):
        super().__init__(env)
        self.scheduled_count = 0
        
    def reset(self, **kwargs):
        self.scheduled_count = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        
        # Override rewards completely
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            reward = 1000.0
            self.scheduled_count += 1
        elif info.get('action_valid', False):
            reward = 1.0
        else:
            reward = -0.1
        
        # Huge completion bonus
        if done or truncated:
            total = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            rate = self.scheduled_count / total if total > 0 else 0
            if rate >= 1.0:
                reward += 10000.0
            elif rate >= 0.9:
                reward += 5000.0
            elif rate >= 0.8:
                reward += 2000.0
        
        return obs, reward, done, truncated, info


def train_to_100(stage_name):
    print(f"\nTraining {stage_name} to 100%...")
    
    # Create wrapped environment
    base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = MaxCompletionWrapper(base_env)
    env = DummyVecEnv([lambda: env])
    
    # Create model with high exploration
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-3,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        ent_coef=0.5,  # Very high exploration
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0
    )
    
    # Train in chunks and check performance
    best_rate = 0
    best_model = None
    
    for i in range(10):  # 10 chunks of 20k timesteps
        model.learn(total_timesteps=20000)
        
        # Evaluate
        eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        scheduled_total = 0
        possible_total = 0
        
        for _ in range(5):
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
            scheduled_total += scheduled
            possible_total += total
        
        rate = scheduled_total / possible_total if possible_total > 0 else 0
        print(f"  Chunk {i+1}: {rate:.1%} completion")
        
        if rate > best_rate:
            best_rate = rate
            best_model = model
            
            # Save if good
            if rate >= 0.9:
                save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/final_push_models"
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"{stage_name}_{int(rate*100)}.zip")
                model.save(model_path)
                print(f"  Saved at {rate:.1%}: {model_path}")
        
        if rate >= 1.0:
            print(f"  ACHIEVED 100%!")
            break
    
    return best_rate


def main():
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    
    print("FINAL PUSH FOR 100% COMPLETION")
    print("=" * 50)
    
    results = {'toy_easy': 1.0}
    
    for stage in stages:
        rate = train_to_100(stage)
        results[stage] = rate
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    all_100 = True
    for stage, rate in results.items():
        status = "✓ 100%!" if rate >= 1.0 else f"{rate:.1%}"
        print(f"{stage}: {status}")
        if rate < 1.0:
            all_100 = False
    
    if all_100:
        print("\n✓ ALL TOYS ACHIEVED 100%!")
        print("Ready to proceed to the next phase!")
    else:
        print("\n✗ Some stages need more work")


if __name__ == "__main__":
    main()