"""
Debug the PPO model to see what's happening with scheduling.
"""

import numpy as np
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

def debug_model():
    print("=== Debugging PPO Model ===")
    
    # Load model
    model = PPO.load("app/models/full_production/final_model.zip")
    print("Model loaded successfully")
    
    # Create environment
    env = FullProductionEnv(
        n_machines=152,
        n_jobs=500,
        state_compression="hierarchical",
        use_break_constraints=True,
        use_holiday_constraints=True,
        seed=42
    )
    
    # Reset and check initial state
    obs, info = env.reset()
    print(f"\nEnvironment reset:")
    print(f"- Number of jobs: {len(env.jobs)}")
    print(f"- Number of machines: {env.n_machines}")
    print(f"- Observation shape: {obs.shape}")
    print(f"- Current time: {env.current_time}")
    
    # Run a few steps to see what's happening
    print("\n=== Running first 10 steps ===")
    for step in range(10):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        print(f"\nStep {step + 1}:")
        print(f"- Action: {action}")
        print(f"- Machine selected: {action % env.n_machines}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"- Reward: {reward:.2f}")
        print(f"- Current time: {env.current_time:.2f}")
        print(f"- Info: {info}")
        
        if terminated or truncated:
            print(f"- Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Check if jobs are being scheduled
    print("\n=== Checking job scheduling ===")
    print(f"Jobs in environment: {len(env.jobs)}")
    
    # Check job attributes
    if env.jobs:
        print("\nFirst job example:")
        job = env.jobs[0]
        print(f"- Job keys: {list(job.keys())}")
        for key, value in job.items():
            print(f"  - {key}: {value}")
    
    # Check if there's a tracking mechanism
    scheduled_count = 0
    for i, job in enumerate(env.jobs):
        if hasattr(job, 'scheduled') and job.get('scheduled', False):
            scheduled_count += 1
        elif hasattr(job, 'scheduled_time') and job.get('scheduled_time', -1) >= 0:
            scheduled_count += 1
    
    print(f"\nScheduled jobs: {scheduled_count}/{len(env.jobs)}")
    
    # Check machine loads
    print("\n=== Machine status ===")
    if hasattr(env, 'machine_loads'):
        active_machines = sum(1 for load in env.machine_loads if load > 0)
        print(f"Active machines: {active_machines}/{env.n_machines}")
        print(f"Machine loads (first 10): {env.machine_loads[:10]}")

if __name__ == "__main__":
    debug_model()