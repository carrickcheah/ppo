"""
Final push for Phase 2 - focused training to beat baseline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.training.train_curriculum import create_env_phase2


def evaluate_model(model, n_episodes=5):
    """Quick evaluation of model performance."""
    eval_env = create_env_phase2(seed=123)
    
    makespans = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        
        base_env = eval_env.unwrapped if hasattr(eval_env, 'unwrapped') else eval_env
        if hasattr(base_env, 'episode_makespan'):
            makespans.append(base_env.episode_makespan)
    
    return np.mean(makespans) if makespans else 0


def main():
    print("\n" + "="*60)
    print("PHASE 2 FINAL PUSH - BEAT THE BASELINE!")
    print("="*60)
    print("Target: Beat 19.4h random baseline with break constraints")
    
    # Load Phase 2 model
    phase2_path = "models/curriculum/phase2_with_breaks/final_model"
    if not Path(phase2_path + ".zip").exists():
        print("ERROR: Phase 2 model not found!")
        return
    
    # Create environments
    print("\nSetting up training environment...")
    env = make_vec_env(
        create_env_phase2,
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'seed': 42}
    )
    
    # Load model
    print("Loading Phase 2 model...")
    model = PPO.load(phase2_path, env=env)
    
    # Initial evaluation
    print("\nInitial evaluation...")
    initial_makespan = evaluate_model(model, n_episodes=3)
    print(f"Current makespan: {initial_makespan:.2f}h")
    print(f"Target: < 19.4h (random baseline)")
    print(f"Gap to close: {initial_makespan - 19.4:.2f}h")
    
    # Aggressive hyperparameters for final push
    model.learning_rate = 1e-4  # Higher learning rate
    model.ent_coef = 0.01       # More exploration
    
    print("\nFinal push configuration:")
    print(f"- Learning rate: {model.learning_rate} (increased)")
    print(f"- Entropy coefficient: {model.ent_coef} (increased)")
    print("- Training timesteps: 500,000")
    print("- Strategy: Aggressive exploration to find better solutions")
    
    # Training loop with frequent evaluation
    print("\nStarting final push training...")
    print("-" * 60)
    
    start_time = datetime.now()
    best_makespan = initial_makespan
    
    for phase in range(5):  # 5 phases of 100k steps each
        print(f"\nPhase {phase + 1}/5...")
        
        # Train
        model.learn(
            total_timesteps=100000,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Evaluate
        current_makespan = evaluate_model(model, n_episodes=5)
        print(f"Makespan after phase {phase + 1}: {current_makespan:.2f}h")
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            print(f"★ NEW BEST: {best_makespan:.2f}h")
            
            # Save if beating baseline
            if best_makespan < 19.4:
                print(f"✓ BEATING BASELINE! Saving model...")
                model.save("models/curriculum/phase2_final/best_model")
        
        # Adjust learning rate if stuck
        if phase > 2 and best_makespan > 19.4:
            model.learning_rate *= 0.5
            print(f"Reducing learning rate to {model.learning_rate}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL PUSH COMPLETE")
    print("="*60)
    
    final_makespan = evaluate_model(model, n_episodes=10)
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"Training time: {training_time:.1f} minutes")
    print(f"Initial makespan: {initial_makespan:.2f}h")
    print(f"Final makespan: {final_makespan:.2f}h")
    print(f"Best achieved: {best_makespan:.2f}h")
    
    # Save final model
    Path("models/curriculum/phase2_final").mkdir(parents=True, exist_ok=True)
    model.save("models/curriculum/phase2_final/final_model")
    
    print("\nRESULTS:")
    if final_makespan < 19.4:
        improvement = (19.4 - final_makespan) / 19.4 * 100
        print(f"✓ SUCCESS! Beat baseline by {improvement:.1f}%")
        print(f"  - Random baseline: 19.4h")
        print(f"  - Curriculum Phase 2: {final_makespan:.2f}h")
    else:
        print(f"⚠ Did not beat baseline")
        print(f"  - Random baseline: 19.4h")
        print(f"  - Curriculum Phase 2: {final_makespan:.2f}h")
        print(f"  - Gap: {final_makespan - 19.4:.2f}h")
    
    print("\nCOMPARISON:")
    print(f"- Phase 1 (no breaks): 16.2h")
    print(f"- Phase 2 (with breaks): {final_makespan:.2f}h")
    print(f"- Break penalty: {(final_makespan - 16.2) / 16.2 * 100:.1f}%")
    
    print("\nNEXT STEPS:")
    if final_makespan < 19.4:
        print("1. Phase 2 successful! Ready for Phase 3 (holidays)")
        print("2. Consider scaling to 152 machines")
    else:
        print("1. Consider different reward shaping")
        print("2. Try curriculum with smaller break windows first")
        print("3. Experiment with different network architectures")


if __name__ == "__main__":
    main()