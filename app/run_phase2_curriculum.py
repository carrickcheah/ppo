"""
Run Phase 2 of curriculum learning - adding break constraints.
This assumes Phase 1 is already complete.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import yaml
import numpy as np
from datetime import datetime
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.environments.scaled_production_env import ScaledProductionEnv
from src.training.train_curriculum import CurriculumTrainingCallback, create_env_phase2


def main():
    print("\n" + "="*60)
    print("CURRICULUM LEARNING - PHASE 2: WITH BREAK CONSTRAINTS")
    print("="*60)
    print("Loading Phase 1 model and adding break constraints")
    
    # Check Phase 1 model exists
    phase1_model_path = "models/curriculum/phase1_no_breaks/final_model"
    if not Path(phase1_model_path + ".zip").exists():
        print(f"ERROR: Phase 1 model not found at {phase1_model_path}")
        print("Please run Phase 1 training first!")
        return
    
    # Load configuration
    config_path = Path("configs/scaled_production_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load Phase 1 model
    print(f"\nLoading Phase 1 model from {phase1_model_path}...")
    phase1_model = PPO.load(phase1_model_path)
    
    # Create Phase 2 environments (with breaks)
    print("\nCreating Phase 2 environments with break constraints...")
    env = make_vec_env(
        create_env_phase2,
        n_envs=config['training']['n_envs'],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'seed': config['training']['seed']}
    )
    
    # Create evaluation environment
    eval_env = create_env_phase2(seed=config['training']['seed'] + 1000)
    
    # Set new environment for the model
    phase1_model.set_env(env)
    
    # Reduce learning rate for fine-tuning
    phase1_model.learning_rate = config['training']['learning_rate'] * 0.5
    print(f"Reduced learning rate to {phase1_model.learning_rate} for fine-tuning")
    
    # Create callback
    callback = CurriculumTrainingCallback("Phase 2: With Breaks")
    
    # Train Phase 2
    print(f"\nTraining Phase 2 for {config['training']['total_timesteps']:,} timesteps...")
    print("This will fine-tune the Phase 1 model to handle break constraints")
    print("-"*60)
    
    start_time = datetime.now()
    
    phase1_model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=False  # Continue from Phase 1
    )
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nPhase 2 completed in {training_time:.1f} minutes")
    
    # Save Phase 2 model
    phase2_path = "models/curriculum/phase2_with_breaks/final_model"
    Path(phase2_path).parent.mkdir(parents=True, exist_ok=True)
    phase1_model.save(phase2_path)
    print(f"Phase 2 model saved to: {phase2_path}")
    
    # Quick evaluation
    print("\nEvaluating Phase 2 performance...")
    total_reward = 0
    makespans = []
    
    for i in range(5):
        obs = eval_env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = phase1_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        if hasattr(eval_env.unwrapped, 'episode_makespan'):
            makespans.append(eval_env.unwrapped.episode_makespan)
    
    print(f"\nPhase 2 Results (5 episodes):")
    print(f"Average reward: {total_reward/5:.1f}")
    if makespans:
        print(f"Average makespan: {np.mean(makespans):.1f}h")
    
    # Save results
    results = {
        'phase': 'Phase 2: With Breaks',
        'training_time_minutes': training_time,
        'avg_reward': total_reward/5,
        'avg_makespan': np.mean(makespans) if makespans else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = "logs/curriculum/phase2_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("The model can now handle break constraints effectively")
    print("="*60)


if __name__ == "__main__":
    main()