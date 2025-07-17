"""
Curriculum learning training script for scaled production environment.
Implements phased training approach to gradually introduce break constraints.

Phase 1: Train without breaks to establish baseline
Phase 2: Add daily breaks (tea, lunch, dinner)
Phase 3: Add weekly breaks and holidays
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
from datetime import datetime
import json
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from src.environments.scaled_production_env import ScaledProductionEnv
from src.training.train_scaled_production import ScaledProductionCallback, evaluate_baselines


class CurriculumTrainingCallback(BaseCallback):
    """Custom callback for tracking curriculum learning progress."""
    
    def __init__(self, phase_name: str, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.phase_name = phase_name
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_makespans = []
        self.episode_utilizations = []
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    self.episode_rewards.append(info["episode"]["r"])
                    
                    # Extract custom metrics if available
                    if "makespan" in info:
                        self.episode_makespans.append(info["makespan"])
                    if "avg_utilization" in info:
                        self.episode_utilizations.append(info["avg_utilization"])
                    
                    if self.episode_count % 10 == 0:
                        elapsed = time.time() - self.start_time
                        mean_reward = np.mean(self.episode_rewards[-50:])
                        
                        print(f"\n[{self.phase_name} - Episode {self.episode_count}] "
                              f"Time: {elapsed/60:.1f}min")
                        print(f"  Mean reward: {mean_reward:.1f}")
                        
                        if self.episode_makespans:
                            mean_makespan = np.mean(self.episode_makespans[-50:])
                            print(f"  Mean makespan: {mean_makespan:.1f}h")
                        
                        if self.episode_utilizations:
                            mean_util = np.mean(self.episode_utilizations[-50:])
                            print(f"  Mean utilization: {mean_util:.2%}")
        
        return True


def create_env_phase1(seed=None):
    """Create Phase 1 environment - no break constraints."""
    env = ScaledProductionEnv(
        n_machines=40,
        max_episode_steps=1000,
        max_valid_actions=100,
        use_break_constraints=False,  # No breaks in Phase 1
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=seed
    )
    return Monitor(env)


def create_env_phase2(seed=None):
    """Create Phase 2 environment - with break constraints."""
    env = ScaledProductionEnv(
        n_machines=40,
        max_episode_steps=1000,
        max_valid_actions=100,
        use_break_constraints=True,  # Enable breaks in Phase 2
        data_file='data/large_production_data.json',
        snapshot_file='data/production_snapshot_latest.json',
        seed=seed
    )
    return Monitor(env)


def train_phase(phase_name: str, 
                create_env_fn,
                config: dict,
                pretrained_model: PPO = None,
                save_path: str = None) -> PPO:
    """
    Train a single phase of curriculum learning.
    
    Args:
        phase_name: Name of the phase (e.g., "Phase 1: No Breaks")
        create_env_fn: Function to create the environment
        config: Training configuration
        pretrained_model: Model from previous phase for transfer learning
        save_path: Path to save the trained model
        
    Returns:
        Trained PPO model
    """
    print("\n" + "="*60)
    print(f"CURRICULUM LEARNING - {phase_name}")
    print("="*60)
    
    # Create environments
    print("Creating training environments...")
    env = make_vec_env(
        create_env_fn,
        n_envs=config['training']['n_envs'],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'seed': config['training']['seed']}
    )
    
    # Create evaluation environment
    eval_env = create_env_fn(seed=config['training']['seed'] + 1000)
    
    # Initialize or load PPO model
    if pretrained_model is None:
        print(f"\nInitializing new PPO model for {phase_name}...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config['training']['learning_rate'],
            n_steps=config['training']['n_steps'],
            batch_size=config['training']['batch_size'],
            n_epochs=config['training']['n_epochs'],
            gamma=config['training']['gamma'],
            gae_lambda=config['training']['gae_lambda'],
            clip_range=config['training']['clip_range'],
            ent_coef=config['training']['ent_coef'],
            vf_coef=config['training']['vf_coef'],
            max_grad_norm=config['training']['max_grad_norm'],
            policy_kwargs=dict(
                net_arch=dict(
                    pi=config['training']['network_arch'],
                    vf=config['training']['network_arch']
                )
            ),
            verbose=0,
            seed=config['training']['seed']
        )
    else:
        print(f"\nLoading pretrained model for {phase_name}...")
        # Set the new environment
        model = pretrained_model
        model.set_env(env)
        # Optionally reduce learning rate for fine-tuning
        model.learning_rate = config['training']['learning_rate'] * 0.5
    
    # Callbacks
    callback = CurriculumTrainingCallback(phase_name)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'models/curriculum/{phase_name.lower().replace(" ", "_").replace(":", "")}/best/',
        log_path=f'logs/curriculum/{phase_name.lower().replace(" ", "_").replace(":", "")}/eval/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f'models/curriculum/{phase_name.lower().replace(" ", "_").replace(":", "")}/checkpoints/',
        name_prefix=f'{phase_name.lower().replace(" ", "_").replace(":", "")}_checkpoint'
    )
    
    # Train
    print(f"\nTraining for {config['training']['total_timesteps']:,} timesteps...")
    print(f"Using {config['training']['n_envs']} parallel environments")
    print(f"Network architecture: {config['training']['network_arch']}")
    print("-"*60)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[callback, eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False if pretrained_model else True
    )
    
    training_time = time.time() - start_time
    print(f"\n{phase_name} completed in {training_time/60:.1f} minutes")
    
    # Save final model
    if save_path:
        model.save(save_path)
        print(f"Model saved to: {save_path}")
    
    # Evaluate
    print(f"\nEvaluating {phase_name} performance...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.1f} (+/- {std_reward:.1f})")
    
    return model


def evaluate_phase_performance(model: PPO, phase_name: str, create_env_fn, n_episodes: int = 10):
    """Evaluate detailed performance metrics for a phase."""
    print(f"\nDetailed evaluation for {phase_name}...")
    
    env = create_env_fn(seed=123)
    makespans = []
    utilizations = []
    setup_ratios = []
    rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        if hasattr(env, 'episode_makespan') and env.episode_makespan > 0:
            makespans.append(env.episode_makespan)
            if hasattr(env, 'machine_utilization'):
                utilizations.append(np.mean(env.machine_utilization))
            if 'setup_ratio' in info:
                setup_ratios.append(info['setup_ratio'])
        rewards.append(total_reward)
    
    results = {
        'mean_reward': np.mean(rewards),
        'mean_makespan': np.mean(makespans) if makespans else 0,
        'mean_utilization': np.mean(utilizations) if utilizations else 0,
        'mean_setup_ratio': np.mean(setup_ratios) if setup_ratios else 0,
        'completion_rate': len(makespans) / n_episodes
    }
    
    print(f"\n{phase_name} Results:")
    print(f"  Mean reward: {results['mean_reward']:.1f}")
    print(f"  Mean makespan: {results['mean_makespan']:.1f}h")
    print(f"  Mean utilization: {results['mean_utilization']:.2%}")
    print(f"  Setup time ratio: {results['mean_setup_ratio']:.2%}")
    print(f"  Completion rate: {results['completion_rate']:.1%}")
    
    return results


def main():
    print("\n" + "="*60)
    print("CURRICULUM LEARNING FOR PRODUCTION SCHEDULING")
    print("="*60)
    print("Phase 1: Train without break constraints")
    print("Phase 2: Add break constraints with transfer learning")
    print("Phase 3: Future - Add holidays and special constraints")
    
    # Load configuration
    config_path = Path("configs/scaled_production_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Store results
    all_results = {}
    
    # Phase 1: Train without breaks
    print("\n" + "-"*60)
    print("STARTING PHASE 1: NO BREAK CONSTRAINTS")
    print("-"*60)
    
    phase1_model = train_phase(
        phase_name="Phase 1: No Breaks",
        create_env_fn=create_env_phase1,
        config=config,
        pretrained_model=None,
        save_path="models/curriculum/phase1_no_breaks/final_model"
    )
    
    # Evaluate Phase 1
    phase1_results = evaluate_phase_performance(
        phase1_model, "Phase 1", create_env_phase1
    )
    all_results['phase1'] = phase1_results
    
    # Compare with baselines
    print("\nComparing Phase 1 with baselines...")
    baseline_results = evaluate_baselines(lambda: create_env_phase1())
    all_results['phase1_baselines'] = baseline_results
    
    # Check if Phase 1 beats baselines
    best_baseline_makespan = min(
        b['mean_makespan'] for b in baseline_results.values()
    )
    phase1_improvement = (1 - phase1_results['mean_makespan']/best_baseline_makespan) * 100
    
    print(f"\nPhase 1 improvement over best baseline: {phase1_improvement:.1f}%")
    
    if phase1_improvement > 0:
        print("✓ Phase 1 successful! Proceeding to Phase 2...")
        
        # Phase 2: Add break constraints
        print("\n" + "-"*60)
        print("STARTING PHASE 2: WITH BREAK CONSTRAINTS")
        print("-"*60)
        
        phase2_model = train_phase(
            phase_name="Phase 2: With Breaks",
            create_env_fn=create_env_phase2,
            config=config,
            pretrained_model=phase1_model,  # Transfer learning
            save_path="models/curriculum/phase2_with_breaks/final_model"
        )
        
        # Evaluate Phase 2
        phase2_results = evaluate_phase_performance(
            phase2_model, "Phase 2", create_env_phase2
        )
        all_results['phase2'] = phase2_results
        
        # Compare Phase 1 vs Phase 2
        makespan_increase = (phase2_results['mean_makespan'] - phase1_results['mean_makespan']) / phase1_results['mean_makespan'] * 100
        print(f"\nMakespan increase due to breaks: {makespan_increase:.1f}%")
        
    else:
        print("✗ Phase 1 did not beat baselines. Need to tune hyperparameters.")
    
    # Save all results
    results_path = "logs/curriculum/curriculum_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'phases': all_results,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("CURRICULUM LEARNING SUMMARY")
    print("="*60)
    
    if 'phase1' in all_results:
        print(f"Phase 1 (No Breaks): {all_results['phase1']['mean_makespan']:.1f}h makespan")
    
    if 'phase2' in all_results:
        print(f"Phase 2 (With Breaks): {all_results['phase2']['mean_makespan']:.1f}h makespan")
    
    print("\nNext steps:")
    if phase1_improvement <= 0:
        print("- Tune hyperparameters for Phase 1")
        print("- Consider different reward shaping")
        print("- Try larger network architecture")
    else:
        print("- Monitor Phase 2 performance")
        print("- Prepare Phase 3 with holidays")
        print("- Consider production deployment")


if __name__ == "__main__":
    main()