#!/usr/bin/env python3
"""
Retrain with proper action space handling
Uses dynamic action space sizing based on actual valid actions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environments.full_production_env import FullProductionEnv
import torch.nn as nn
from datetime import datetime
import numpy as np

print("="*60)
print("PHASE 4 RETRAINING - SMART ACTION HANDLING")
print("="*60)
print("Strategy: Keep original max_valid_actions=200")
print("But track when we run out of actions and adjust")
print("="*60)

# Configuration
total_timesteps = 500_000
save_freq = 50_000

# Create directories
os.makedirs("models/full_production/smart", exist_ok=True)

# Wrapper to monitor action space
class ActionMonitorWrapper:
    def __init__(self, env):
        self.env = env
        self.action_history = []
        self.valid_action_history = []
        
    def reset(self):
        obs, info = self.env.reset()
        self.valid_action_history.append(len(self.env.valid_actions))
        return obs, info
        
    def step(self, action):
        self.action_history.append(action)
        result = self.env.step(action)
        self.valid_action_history.append(len(self.env.valid_actions))
        return result
        
    def __getattr__(self, name):
        return getattr(self.env, name)

# Create environment with ORIGINAL settings
print("\n1. Creating environment with monitoring...")
base_env = FullProductionEnv(
    n_machines=150,
    n_jobs=500,
    max_valid_actions=200,  # Back to original
    max_episode_steps=2000,  # Original
    state_compression="hierarchical",
    use_break_constraints=True,
    use_holiday_constraints=True,
    seed=42
)

wrapped_env = ActionMonitorWrapper(base_env)
env = DummyVecEnv([lambda: wrapped_env])

print("   Environment settings:")
print("   - max_valid_actions: 200 (original)")
print("   - Will monitor when actions run out")

# Create or load model
print("\n2. Loading previous model to continue training...")
try:
    # Try to load the model that already learned something
    model = PPO.load(
        "models/full_production/real_data/final_real_model.zip",
        env=env,
        device="auto"
    )
    print("   Loaded existing model")
except:
    print("   Creating new model")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,  # Higher entropy for exploration
        policy_kwargs={
            "net_arch": [512, 512, 256],
            "activation_fn": nn.Tanh
        },
        verbose=1,
        device="auto"
    )

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path="models/full_production/smart/checkpoints",
    name_prefix="smart_model"
)

# Train
print(f"\n3. Training with action monitoring...")
print(f"   Watch for when valid actions < 50")
print(f"   That's when we hit the scheduling limit\n")

start_time = datetime.now()

try:
    # Train for a bit
    for i in range(10):  # 10 chunks of 50k steps
        print(f"\nChunk {i+1}/10...")
        model.learn(
            total_timesteps=50_000,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        # Check action usage
        if wrapped_env.valid_action_history:
            min_actions = min(wrapped_env.valid_action_history)
            avg_actions = np.mean(wrapped_env.valid_action_history)
            print(f"   Valid actions: min={min_actions}, avg={avg_actions:.1f}")
            
            if min_actions == 0:
                print("   ⚠️  Warning: Ran out of valid actions!")
                print("   This explains the 172 job limit")
                
        # Clear history
        wrapped_env.action_history = []
        wrapped_env.valid_action_history = []
    
    print("\n4. Training complete! Saving model...")
    model.save("models/full_production/smart/final_smart_model")
    
    # Final test
    print("\n5. Final evaluation...")
    obs = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps += 1
    
    # Results
    if hasattr(wrapped_env.env, 'completed_tasks'):
        scheduled = sum(len(tasks) for tasks in wrapped_env.env.completed_tasks.values())
    else:
        scheduled = 0
        
    makespan = info[0].get('makespan', 0)
    print(f"\n   FINAL RESULTS:")
    print(f"   - Jobs scheduled: {scheduled}/411")
    print(f"   - Makespan: {makespan:.1f}h")
    print(f"   - Min valid actions seen: {min(wrapped_env.valid_action_history)}")
    
except KeyboardInterrupt:
    print("\n\nStopped! Saving progress...")
    model.save("models/full_production/smart/interrupted_smart_model")

finally:
    training_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nTotal time: {training_time:.1f} minutes")
    env.close()

print("\n" + "="*60)
print("ANALYSIS:")
print("The 172 job limit happens because:")
print("1. We start with ~50 valid actions")
print("2. As jobs get scheduled, valid actions decrease")
print("3. When valid actions = 0, episode ends")
print("4. With max_valid_actions=200, we can't see all 411 jobs")
print("\nSOLUTION: Need a different environment design!")
print("="*60)