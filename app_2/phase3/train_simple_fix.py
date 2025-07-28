"""
Simple fix: Just reduce late penalty and train longer
This is all we need to achieve 80%
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


def evaluate_model(model, stage_name, n_episodes=20):
    """Evaluate model performance."""
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = Monitor(env)
    
    results = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < env.env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            steps += 1
        
        scheduled = len(env.env.scheduled_jobs)
        total = env.env.total_tasks
        rate = scheduled / total
        results.append(rate)
        
        if ep < 5:  # Show first 5 episodes
            print(f"  Episode {ep+1}: {scheduled}/{total} = {rate:.1%}")
    
    avg_rate = np.mean(results)
    std_rate = np.std(results)
    
    return {
        'avg_rate': avg_rate,
        'std_rate': std_rate,
        'min_rate': np.min(results),
        'max_rate': np.max(results)
    }


def train_simple_fix(stage_name, timesteps=1000000):
    """Train with simple fix: reduced late penalty + longer training."""
    
    print(f"\n{'='*60}")
    print(f"Training {stage_name} with SIMPLE FIX")
    print(f"Changes: late_penalty -5.0 → -2.0, timesteps 500k → 1M")
    print(f"{'='*60}\n")
    
    # Create environment
    def make_env():
        env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Use EXACT SAME hyperparameters that got 56.2%
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,  # Same as successful run
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Checkpoints
    checkpoint_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/simple_fix/{stage_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="checkpoint"
    )
    
    # Train
    print(f"Starting training for {timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    print(f"\nEvaluating {stage_name}...")
    results = evaluate_model(model, stage_name)
    
    print(f"\nRESULTS for {stage_name}:")
    print(f"  Average: {results['avg_rate']:.1%} (±{results['std_rate']:.1%})")
    print(f"  Min/Max: {results['min_rate']:.1%} / {results['max_rate']:.1%}")
    print(f"  Training time: {training_time/60:.1f} minutes")
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_simple_fix"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_simple_fix.zip")
    model.save(model_path)
    
    # Save results
    results_data = {
        'stage': stage_name,
        'avg_rate': results['avg_rate'],
        'std_rate': results['std_rate'],
        'min_rate': results['min_rate'],
        'max_rate': results['max_rate'],
        'training_time_min': training_time/60,
        'timesteps': timesteps,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, f"{stage_name}_results.json"), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return results['avg_rate']


def main():
    """Train all stages with simple fix."""
    
    print("\nSIMPLE FIX TRAINING")
    print("=" * 60)
    print("Approach: Just reduce late penalty from -5 to -2 and train longer")
    print("No complex wrappers, no action masking, no reward engineering")
    print("Just the simple fix that should work!")
    print("-" * 60)
    
    baseline = {
        'toy_normal': 0.562,
        'toy_hard': 0.30,
        'toy_multi': 0.364
    }
    
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    results = {}
    
    for stage in stages:
        print(f"\nBaseline for {stage}: {baseline[stage]:.1%}")
        rate = train_simple_fix(stage)
        results[stage] = rate
        
        if rate >= 0.8:
            print(f"\n✓ {stage} ACHIEVED 80% TARGET! ({rate:.1%})")
        else:
            improvement = rate - baseline[stage]
            print(f"\n{stage}: {rate:.1%} (improved {improvement:+.1%} from baseline)")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS - SIMPLE FIX")
    print("="*60)
    
    all_pass = True
    for stage in stages:
        old = baseline[stage]
        new = results[stage]
        improvement = new - old
        
        status = "✓" if new >= 0.8 else "✗"
        print(f"{status} {stage:12} {old:.1%} → {new:.1%} ({improvement:+.1%})")
        
        if new < 0.8:
            all_pass = False
    
    if all_pass:
        print("\n✓ SUCCESS! All stages achieved 80% with just simple penalty reduction!")
    else:
        print("\nPartial success. May need slightly lower penalty or more training.")


if __name__ == "__main__":
    main()