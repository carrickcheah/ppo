"""
Train and compare boolean importance models (constrained vs unconstrained).
Tests if boolean is_important flag is clearer for PPO than 1-5 priority.
"""

import sys
from pathlib import Path
import numpy as np
import time
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.environments.medium_env_boolean import MediumBooleanSchedulingEnv
from src.environments.medium_env_boolean_unconstrained import MediumBooleanUnconstrainedSchedulingEnv


class BooleanProgressCallback(BaseCallback):
    """Track progress with focus on importance violations."""
    
    def __init__(self, env_name, check_freq=1000):
        super().__init__()
        self.env_name = env_name
        self.check_freq = check_freq
        self.episode_count = 0
        self.recent_rewards = []
        self.recent_violations = []
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    self.recent_rewards.append(info["episode"]["r"])
                    
                    if "importance_violations" in info:
                        self.recent_violations.append(info["importance_violations"])
                    
                    if self.episode_count % 10 == 0:
                        elapsed = time.time() - self.start_time
                        mean_reward = np.mean(self.recent_rewards[-50:])
                        
                        print(f"\n[{self.env_name}] Episode {self.episode_count} "
                              f"({elapsed/60:.1f} min)")
                        print(f"  Mean reward: {mean_reward:.1f}")
                        
                        if self.recent_violations:
                            mean_violations = np.mean(self.recent_violations[-50:])
                            print(f"  Importance violations: {mean_violations:.1f}/episode")
        
        return True


def create_constrained_env(seed=None):
    """Create constrained boolean environment."""
    return Monitor(MediumBooleanSchedulingEnv(seed=seed))


def create_unconstrained_env(seed=None):
    """Create unconstrained boolean environment."""
    return Monitor(MediumBooleanUnconstrainedSchedulingEnv(seed=seed))


def train_model(env_fn, model_name, timesteps=100000):
    """Train a model with given environment."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create environments
    env = make_vec_env(env_fn, n_envs=4, vec_env_cls=SubprocVecEnv)
    
    # Configure PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=0
    )
    
    # Train
    callback = BooleanProgressCallback(model_name)
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=False
    )
    
    training_time = time.time() - start_time
    print(f"\n{model_name} training completed in {training_time/60:.1f} minutes")
    
    # Save model
    model_path = f"./models/boolean/{model_name.lower().replace(' ', '_')}"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    
    return model, training_time


def evaluate_model(model, env_class, model_name, n_episodes=10):
    """Evaluate a trained model."""
    print(f"\nEvaluating {model_name}...")
    
    env = env_class(seed=42)
    
    results = {
        'rewards': [],
        'makespans': [],
        'completions': [],
        'violations': [],
        'urgency_justified': []
    }
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Collect metrics
        completed = sum(len(c) for c in env.completed_tasks.values())
        results['rewards'].append(total_reward)
        results['makespans'].append(env.episode_makespan)
        results['completions'].append(completed / env.n_jobs)
        
        if hasattr(env, 'importance_violations'):
            results['violations'].append(env.importance_violations)
            results['urgency_justified'].append(env.urgency_justified)
        else:
            results['violations'].append(0)
            results['urgency_justified'].append(0)
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(results['rewards']),
        'mean_makespan': np.mean(results['makespans']),
        'mean_completion': np.mean(results['completions']),
        'mean_violations': np.mean(results['violations']),
        'mean_urgency_justified': np.mean(results['urgency_justified']),
        'efficiency': 26.6 / np.mean(results['makespans']) if np.mean(results['makespans']) > 0 else 0
    }
    
    return stats


def main():
    print("="*60)
    print("BOOLEAN IMPORTANCE SYSTEM - TRAINING & COMPARISON")
    print("="*60)
    print("Testing is_important (True/False) vs priority (1-5)")
    
    # Training parameters
    timesteps = 100_000
    
    # Train constrained model
    constrained_model, constrained_time = train_model(
        create_constrained_env,
        "Boolean Constrained",
        timesteps
    )
    
    # Train unconstrained model
    unconstrained_model, unconstrained_time = train_model(
        create_unconstrained_env,
        "Boolean Unconstrained",
        timesteps
    )
    
    # Evaluate both models
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    constrained_stats = evaluate_model(
        constrained_model,
        MediumBooleanSchedulingEnv,
        "Boolean Constrained"
    )
    
    unconstrained_stats = evaluate_model(
        unconstrained_model,
        MediumBooleanUnconstrainedSchedulingEnv,
        "Boolean Unconstrained"
    )
    
    # Compare with previous results
    print("\n" + "="*60)
    print("COMPARISON: BOOLEAN vs PRIORITY SYSTEMS")
    print("="*60)
    
    # Previous results with 1-5 priority
    priority_results = {
        'constrained': {'makespan': 27.9, 'efficiency': 0.953},
        'unconstrained': {'makespan': 27.4, 'efficiency': 0.971}
    }
    
    print(f"\n{'System':<25} {'Constrained':>15} {'Unconstrained':>15}")
    print("-"*55)
    
    # Makespan comparison
    print(f"{'Priority (1-5) Makespan':<25} {priority_results['constrained']['makespan']:>14.1f}h "
          f"{priority_results['unconstrained']['makespan']:>14.1f}h")
    print(f"{'Boolean Makespan':<25} {constrained_stats['mean_makespan']:>14.1f}h "
          f"{unconstrained_stats['mean_makespan']:>14.1f}h")
    
    # Efficiency comparison
    print(f"\n{'Priority (1-5) Efficiency':<25} {priority_results['constrained']['efficiency']:>14.1%} "
          f"{priority_results['unconstrained']['efficiency']:>14.1%}")
    print(f"{'Boolean Efficiency':<25} {constrained_stats['efficiency']:>14.1%} "
          f"{unconstrained_stats['efficiency']:>14.1%}")
    
    # Violations
    print(f"\n{'Boolean Violations':<25} {constrained_stats['mean_violations']:>14.0f} "
          f"{unconstrained_stats['mean_violations']:>14.0f}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS: BOOLEAN SYSTEM ADVANTAGES")
    print("="*60)
    
    boolean_imp = ((priority_results['unconstrained']['makespan'] - unconstrained_stats['mean_makespan']) 
                   / priority_results['unconstrained']['makespan'] * 100)
    
    if abs(boolean_imp) > 1:
        if boolean_imp > 0:
            print(f"✅ Boolean system is {boolean_imp:.1f}% BETTER than 1-5 priority!")
        else:
            print(f"❌ Boolean system is {-boolean_imp:.1f}% worse than 1-5 priority")
    else:
        print("➖ Similar performance between boolean and 1-5 priority systems")
    
    print(f"\nKey insights:")
    print(f"1. Boolean system is simpler (2 states vs 5 levels)")
    print(f"2. Clearer signal for neural network (0 or 1)")
    print(f"3. Violations are more meaningful (important vs not)")
    
    if unconstrained_stats['mean_violations'] > 0:
        violation_rate = unconstrained_stats['mean_violations'] / 172 * 100
        print(f"\nUnconstrained behavior:")
        print(f"- Violated importance in {violation_rate:.1f}% of decisions")
        print(f"- {unconstrained_stats['mean_urgency_justified']:.0f} were urgency-justified")
    
    # Save results - convert numpy types to Python types
    results = {
        'boolean_constrained': {k: float(v) if isinstance(v, np.floating) else v 
                               for k, v in constrained_stats.items()},
        'boolean_unconstrained': {k: float(v) if isinstance(v, np.floating) else v 
                                 for k, v in unconstrained_stats.items()},
        'priority_comparison': priority_results,
        'training_times': {
            'constrained': float(constrained_time / 60),
            'unconstrained': float(unconstrained_time / 60)
        }
    }
    
    with open('./logs/boolean_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: ./logs/boolean_comparison_results.json")


if __name__ == "__main__":
    main()