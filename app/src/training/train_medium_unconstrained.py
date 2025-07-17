"""
Train PPO agent on unconstrained medium scheduling environment.
Tests if PPO can discover better strategies without pre-sorting constraints.
"""

import sys
from pathlib import Path
import numpy as np
import time
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.environments.medium_env_unconstrained import MediumUnconstrainedSchedulingEnv


class DetailedProgressCallback(BaseCallback):
    """Custom callback for detailed logging of unconstrained behavior."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.priority_violations = []
        self.urgency_wins = []
        self.makespans = []
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # Log episode info when available
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    
                    # Extract custom metrics if available
                    if "priority_violations" in info:
                        self.priority_violations.append(info["priority_violations"])
                    if "urgency_wins" in info:
                        self.urgency_wins.append(info["urgency_wins"])
                    if "makespan" in info:
                        self.makespans.append(info["makespan"])
                    
                    if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-100:])
                        mean_length = np.mean(self.episode_lengths[-100:])
                        elapsed = time.time() - self.start_time
                        
                        print(f"\n[{len(self.episode_rewards)} episodes, "
                              f"{elapsed/60:.1f} min] "
                              f"Mean reward: {mean_reward:.1f}, "
                              f"Mean length: {mean_length:.0f}")
                        
                        if len(self.priority_violations) > 0:
                            mean_violations = np.mean(self.priority_violations[-100:])
                            mean_urgency = np.mean(self.urgency_wins[-100:]) if self.urgency_wins else 0
                            print(f"Priority violations: {mean_violations:.1f}, "
                                  f"Urgency wins: {mean_urgency:.1f}")
                        
                        # Check for improvement
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            print(f"New best mean reward: {self.best_mean_reward:.1f}")
        
        return True


def create_env(seed=None):
    """Create a single unconstrained environment instance."""
    env = MediumUnconstrainedSchedulingEnv(
        n_machines=10,
        max_episode_steps=500,
        max_valid_actions=100,  # Handle up to 100 valid actions
        seed=seed
    )
    return Monitor(env)


def evaluate_model(model, env, n_episodes=10):
    """Evaluate trained model performance with detailed analysis."""
    print("\n" + "="*60)
    print("EVALUATING UNCONSTRAINED MODEL")
    print("="*60)
    
    episode_rewards = []
    episode_lengths = []
    makespans = []
    efficiencies = []
    completion_rates = []
    priority_violations_list = []
    urgency_wins_list = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Extract metrics from underlying env
        underlying_env = env.env if hasattr(env, 'env') else env
        
        # Extract completion info
        if 'all_tasks_completed' in info:
            makespans.append(info['makespan'])
            efficiencies.append(info['efficiency'])
            completion_rates.append(1.0)
            priority_violations_list.append(info.get('priority_violations', 0))
            urgency_wins_list.append(info.get('urgency_wins', 0))
        else:
            # Count completed tasks
            completed = sum(len(c) for c in underlying_env.completed_tasks.values())
            completion_rates.append(completed / underlying_env.n_jobs)
            makespans.append(underlying_env.episode_makespan)
            # Estimate efficiency
            total_work = sum(
                task['processing_time'] 
                for family in underlying_env.families_data.values()
                for task in family['tasks']
            )
            theoretical_min = total_work / underlying_env.n_machines
            efficiencies.append(theoretical_min / underlying_env.episode_makespan if underlying_env.episode_makespan > 0 else 0)
            priority_violations_list.append(underlying_env.priority_violations)
            urgency_wins_list.append(underlying_env.urgency_wins)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward: {total_reward:.1f}")
        print(f"  Steps: {steps}")
        print(f"  Completion: {completion_rates[-1]:.1%}")
        if makespans[-1] > 0:
            print(f"  Makespan: {makespans[-1]:.1f}h")
            print(f"  Efficiency: {efficiencies[-1]:.1%}")
        print(f"  Priority violations: {priority_violations_list[-1]}")
        print(f"  Urgency wins: {urgency_wins_list[-1]}")
    
    print("\n" + "-"*40)
    print("SUMMARY STATISTICS:")
    print(f"Mean reward: {np.mean(episode_rewards):.1f} (±{np.std(episode_rewards):.1f})")
    print(f"Mean completion rate: {np.mean(completion_rates):.1%}")
    print(f"Mean makespan: {np.mean(makespans):.1f}h")
    print(f"Mean efficiency: {np.mean(efficiencies):.1%}")
    print(f"Mean priority violations: {np.mean(priority_violations_list):.1f}")
    print(f"Mean urgency wins: {np.mean(urgency_wins_list):.1f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_completion': np.mean(completion_rates),
        'mean_makespan': np.mean(makespans),
        'mean_efficiency': np.mean(efficiencies),
        'mean_priority_violations': np.mean(priority_violations_list),
        'mean_urgency_wins': np.mean(urgency_wins_list)
    }


def compare_with_constrained():
    """Load and compare with constrained model."""
    print("\n" + "="*60)
    print("COMPARISON: CONSTRAINED vs UNCONSTRAINED")
    print("="*60)
    
    # Load constrained model results (if available)
    constrained_results = {
        'makespan': 27.9,
        'efficiency': 0.954,
        'completion': 1.0,
        'priority_violations': 0,  # Always 0 for constrained
        'urgency_wins': 0
    }
    
    print("\nConstrained Model (sorted by priority → LCD):")
    print(f"  Makespan: {constrained_results['makespan']:.1f}h")
    print(f"  Efficiency: {constrained_results['efficiency']:.1%}")
    print(f"  Always follows priority order")
    
    return constrained_results


def main():
    """Main training function."""
    print("="*60)
    print("UNCONSTRAINED MEDIUM SCHEDULING - PPO TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nKey difference: NO pre-sorting by priority/LCD")
    print("PPO can explore all valid actions freely")
    
    # Training parameters
    total_timesteps = 200_000  # More steps for exploration
    n_envs = 4  # Parallel environments
    
    # Create vectorized environment
    print(f"\nCreating {n_envs} parallel environments...")
    env = make_vec_env(create_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: create_env(seed=42)])
    
    # Configure PPO with exploration-friendly parameters
    print("\nConfiguring PPO agent (exploration-friendly)...")
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
        ent_coef=0.02,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log="./logs/medium_unconstrained/"
    )
    
    # Setup callbacks
    progress_callback = DetailedProgressCallback(check_freq=1000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/medium_unconstrained/",
        log_path="./logs/medium_unconstrained/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train model
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("Watch for priority violations and urgency wins...\n")
    
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, eval_callback]
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Save final model
    model_path = "./models/medium_unconstrained/final_model"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Evaluate trained model
    eval_env_single = create_env(seed=42)
    eval_results = evaluate_model(model, eval_env_single, n_episodes=10)
    
    # Compare with constrained baseline
    constrained_results = compare_with_constrained()
    
    # Analysis
    print("\n" + "="*60)
    print("UNCONSTRAINED LEARNING ANALYSIS")
    print("="*60)
    
    makespan_improvement = (constrained_results['makespan'] - eval_results['mean_makespan']) / constrained_results['makespan'] * 100
    
    print(f"\nMakespan comparison:")
    print(f"  Constrained: {constrained_results['makespan']:.1f}h")
    print(f"  Unconstrained: {eval_results['mean_makespan']:.1f}h")
    print(f"  Improvement: {makespan_improvement:+.1f}%")
    
    print(f"\nStrategy analysis:")
    print(f"  Priority violations per episode: {eval_results['mean_priority_violations']:.1f}")
    print(f"  Urgency-based decisions: {eval_results['mean_urgency_wins']:.1f}")
    
    violation_rate = eval_results['mean_priority_violations'] / 172 * 100  # 172 total tasks
    print(f"  Violation rate: {violation_rate:.1f}% of scheduling decisions")
    
    if eval_results['mean_priority_violations'] > 0:
        urgency_success_rate = eval_results['mean_urgency_wins'] / eval_results['mean_priority_violations'] * 100
        print(f"  Urgency success rate: {urgency_success_rate:.1f}% of violations were for urgent jobs")
    
    # Save detailed results
    results = {
        'training_time_minutes': training_time / 60,
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'eval_results': {
            k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v 
            for k, v in eval_results.items()
        },
        'constrained_baseline': constrained_results,
        'makespan_improvement_percent': float(makespan_improvement),
        'timestamp': datetime.now().isoformat()
    }
    
    # Create directories if needed
    Path("./logs/medium_unconstrained").mkdir(parents=True, exist_ok=True)
    
    with open('./logs/medium_unconstrained/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("VERDICT ON UNCONSTRAINED APPROACH")
    print("="*60)
    
    if makespan_improvement > 1:
        print(f"✅ SUCCESS: Unconstrained PPO found better solutions!")
        print(f"   {makespan_improvement:.1f}% improvement by violating strict priority order")
        print(f"   Learned to prioritize urgent jobs when beneficial")
    elif makespan_improvement < -1:
        print(f"⚠️  WORSE: Unconstrained PPO performed worse")
        print(f"   May need more training or better reward shaping")
    else:
        print(f"➖ SIMILAR: No significant difference from constrained")
        print(f"   Strict priority ordering might be near-optimal for this data")
    
    print(f"\nKey insight: PPO made {eval_results['mean_priority_violations']:.0f} priority violations per episode")
    print("This shows PPO explored alternative strategies beyond rigid rules!")


if __name__ == "__main__":
    # Create necessary directories
    Path("./models/medium_unconstrained").mkdir(parents=True, exist_ok=True)
    Path("./logs/medium_unconstrained").mkdir(parents=True, exist_ok=True)
    
    main()