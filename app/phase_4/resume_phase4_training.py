"""
Resume Phase 4 training from checkpoint.
This script loads the checkpoint at 400k steps and continues training to 1M steps.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.environments.full_production_env import FullProductionEnv
from src.training.train_full_production import Phase4Config, make_env, DetailedLoggingCallback, evaluate_baselines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resume_phase4_training():
    """Resume Phase 4 training from checkpoint."""
    config = Phase4Config()
    
    # Checkpoint to resume from
    checkpoint_path = "app/models/full_production/checkpoints/phase4_model_400000_steps.zip"
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return None, None
        
    logger.info("=== Resuming Phase 4 Training ===")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environment
    logger.info(f"Creating {config.n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(config, i) for i in range(config.n_envs)])
    
    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env(config, 9999)])
    
    # Load model from checkpoint
    logger.info("Loading model from checkpoint...")
    model = PPO.load(
        checkpoint_path,
        env=env,
        device="auto"
    )
    
    # Update learning rate (in case we want to use a different schedule)
    model.learning_rate = config.learning_rate
    
    # Evaluate current performance
    logger.info("\nEvaluating checkpoint model performance...")
    
    # Quick evaluation without using evaluate_model due to stats compatibility issues
    eval_env_single = FullProductionEnv(
        n_machines=config.n_machines,
        n_jobs=config.n_jobs,
        state_compression=config.state_compression,
        use_break_constraints=config.use_break_constraints,
        use_holiday_constraints=config.use_holiday_constraints,
        seed=9999
    )
    
    checkpoint_results = []
    for episode in range(3):
        obs, info = eval_env_single.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_single.step(action)
            episode_reward += reward
            steps += 1
            
        checkpoint_results.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': steps,
            'makespan': info.get('makespan', 0),
            'avg_utilization': info.get('avg_utilization', 0)
        })
        
    eval_env_single.close()
    
    checkpoint_results = {
        'avg_reward': np.mean([r['reward'] for r in checkpoint_results]),
        'avg_makespan': np.mean([r['makespan'] for r in checkpoint_results]),
        'avg_utilization': np.mean([r['avg_utilization'] for r in checkpoint_results])
    }
    
    logger.info(f"Checkpoint evaluation results:")
    logger.info(f"  Average reward: {checkpoint_results['avg_reward']:.2f}")
    logger.info(f"  Average makespan: {checkpoint_results['avg_makespan']:.2f}h")
    logger.info(f"  Average utilization: {checkpoint_results['avg_utilization']:.2%}")
    
    # Calculate remaining timesteps
    completed_timesteps = 400_000
    remaining_timesteps = config.total_timesteps - completed_timesteps
    
    logger.info(f"\nTraining progress:")
    logger.info(f"  Completed: {completed_timesteps:,} steps")
    logger.info(f"  Remaining: {remaining_timesteps:,} steps")
    logger.info(f"  Total: {config.total_timesteps:,} steps")
    
    # Set up callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best_model"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=3
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(output_path / "checkpoints"),
        name_prefix="phase4_model"
    )
    
    logging_callback = DetailedLoggingCallback(
        log_dir=str(output_path / "detailed_logs")
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback, logging_callback])
    
    # Resume training
    logger.info(f"\nResuming training for {remaining_timesteps:,} timesteps...")
    start_time = datetime.now()
    
    # IMPORTANT: reset_num_timesteps=False to continue from where we left off
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,  # This ensures we continue from 400k
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save final model
    final_model_path = output_path / "final_model"
    model.save(str(final_model_path))
    logger.info(f"Saved final model to {final_model_path}")
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    
    # Custom evaluation due to stats compatibility
    eval_env_single = FullProductionEnv(
        n_machines=config.n_machines,
        n_jobs=config.n_jobs,
        state_compression=config.state_compression,
        use_break_constraints=config.use_break_constraints,
        use_holiday_constraints=config.use_holiday_constraints,
        seed=9999
    )
    
    final_results_list = []
    for episode in range(5):
        obs, info = eval_env_single.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_single.step(action)
            episode_reward += reward
            steps += 1
            
        final_results_list.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': steps,
            'makespan': info.get('makespan', 0),
            'avg_utilization': info.get('avg_utilization', 0),
            'n_jobs_scheduled': info.get('n_jobs_scheduled', len(eval_env_single.jobs))
        })
        
    eval_env_single.close()
    
    # Calculate completion rate based on jobs scheduled
    total_jobs = config.n_jobs if config.n_jobs else 172  # Default to 172 jobs
    
    final_results = {
        'avg_reward': np.mean([r['reward'] for r in final_results_list]),
        'avg_makespan': np.mean([r['makespan'] for r in final_results_list]),
        'avg_utilization': np.mean([r['avg_utilization'] for r in final_results_list]),
        'avg_completion_rate': np.mean([r['n_jobs_scheduled'] / total_jobs for r in final_results_list])
    }
    
    logger.info(f"Final evaluation results:")
    logger.info(f"  Average reward: {final_results['avg_reward']:.2f}")
    logger.info(f"  Average makespan: {final_results['avg_makespan']:.2f}h")
    logger.info(f"  Average utilization: {final_results['avg_utilization']:.2%}")
    logger.info(f"  Average completion rate: {final_results['avg_completion_rate']:.2%}")
    
    # Save results
    results_data = {
        "config": {
            "n_machines": config.n_machines,
            "n_jobs": config.n_jobs,
            "state_compression": config.state_compression,
            "total_timesteps": config.total_timesteps,
            "learning_rate": config.learning_rate
        },
        "checkpoint_results": checkpoint_results,
        "final_results": final_results,
        "training_time_seconds": training_time,
        "resumed_from_checkpoint": True,
        "checkpoint_timesteps": completed_timesteps
    }
    
    with open(output_path / "training_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
        
    # Compare with baselines
    logger.info("\nComparing with baseline policies...")
    baseline_results = evaluate_baselines(config)
    
    # Summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Scale: {config.n_machines} machines, {config.n_jobs} jobs")
    logger.info(f"Additional training time: {training_time/60:.1f} minutes")
    
    logger.info(f"\nCheckpoint model performance (400k steps):")
    logger.info(f"  Makespan: {checkpoint_results['avg_makespan']:.2f}h")
    # Note: completion rate not available in checkpoint evaluation
    
    logger.info(f"\nFinal model performance (1M steps):")
    logger.info(f"  Makespan: {final_results['avg_makespan']:.2f}h")
    logger.info(f"  Completion rate: {final_results['avg_completion_rate']:.2%}")
    
    improvement = (checkpoint_results['avg_makespan'] - final_results['avg_makespan']) / checkpoint_results['avg_makespan'] * 100
    logger.info(f"\nImprovement from additional training: {improvement:.1f}%")
    
    logger.info(f"\nBaseline comparison:")
    for name, result in baseline_results.items():
        logger.info(f"  {name}: {result['avg_makespan']:.2f}h")
        diff = (result['avg_makespan'] - final_results['avg_makespan']) / result['avg_makespan'] * 100
        logger.info(f"    PPO improvement: {diff:.1f}%")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model, results_data


if __name__ == "__main__":
    model, results = resume_phase4_training()
    
    if results:
        logger.info("\n=== Phase 4 Training Complete ===")
        logger.info("Next steps:")
        logger.info("1. Run: uv run python visualize_phase4_results.py")
        logger.info("2. Check results in: app/models/full_production/training_results.json")
        logger.info("3. Model ready at: app/models/full_production/final_model.zip")