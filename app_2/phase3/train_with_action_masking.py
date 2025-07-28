"""
Train toy stages with MaskablePPO using action masking
Expected to achieve 80%+ completion by only exploring valid actions
"""

import os
import sys
import time
import json
import logging
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed
from phase3.environments.action_masked_env import ToyStageActionMasker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_masked_env(stage_name: str):
    """Create action-masked environment."""
    def _init():
        env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        env = ToyStageActionMasker(env)  # This wraps with action masking
        env = Monitor(env)
        return env
    return _init


def evaluate_masked_model(model, stage_name: str, n_eval_episodes: int = 20):
    """Evaluate model performance."""
    # Use same wrapped environment as training
    eval_env = create_masked_env(stage_name)()
    
    total_scheduled = 0
    total_possible = 0
    episode_rewards = []
    episode_rates = []
    
    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 300:
            # Predict action (MaskablePPO handles masking internally during predict)
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward
            steps += 1
            done = done or truncated
        
        # Get results from the original environment
        # Navigate through wrappers: Monitor -> ToyStageActionMasker -> MultiDiscreteToDiscreteWrapper -> CurriculumEnvironmentTrulyFixed
        if hasattr(eval_env, 'env'):  # Monitor
            inner_env = eval_env.env
            if hasattr(inner_env, 'original_env'):  # ToyStageActionMasker
                original_env = inner_env.original_env
            elif hasattr(inner_env, 'env'):  # MultiDiscreteToDiscreteWrapper
                original_env = inner_env.env
            else:
                original_env = inner_env
        else:
            original_env = eval_env
            
        scheduled = len(original_env.scheduled_jobs) if hasattr(original_env, 'scheduled_jobs') else 0
        total = original_env.total_tasks if hasattr(original_env, 'total_tasks') else 1
        rate = scheduled / total if total > 0 else 0
        
        total_scheduled += scheduled
        total_possible += total
        episode_rewards.append(ep_reward)
        episode_rates.append(rate)
        
        if ep == 0:
            logger.info(f"First episode: {scheduled}/{total} = {rate:.1%}, reward: {ep_reward:.1f}")
    
    avg_rate = total_scheduled / total_possible if total_possible > 0 else 0
    
    return {
        'avg_rate': avg_rate,
        'std_rate': np.std(episode_rates),
        'min_rate': np.min(episode_rates),
        'max_rate': np.max(episode_rates),
        'avg_reward': np.mean(episode_rewards),
        'episodes': n_eval_episodes
    }


def train_stage_with_masking(stage_name: str, timesteps: int = 500000):
    """Train a single stage with action masking."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {stage_name} with ACTION MASKING")
    logger.info(f"Expected: 80%+ completion (currently best: 56.2%)")
    logger.info(f"{'='*60}")
    
    # Create vectorized environment
    env = DummyVecEnv([create_masked_env(stage_name)])
    
    # Create MaskablePPO model
    model = MaskablePPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,  # Same as adjusted hyperparams
        n_steps=512,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.1,  # Balanced exploration
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1
    )
    
    # Setup logging
    log_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/logs/{stage_name}_masked"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    
    # Checkpoints
    checkpoint_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/checkpoints/masked/{stage_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="masked_checkpoint"
    )
    
    # Training
    start_time = time.time()
    logger.info(f"Starting MASKED training for {timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    
    training_time = time.time() - start_time
    
    # Evaluation
    logger.info(f"\nEvaluating {stage_name} with masking...")
    eval_results = evaluate_masked_model(model, stage_name)
    
    logger.info(f"\nRESULTS for {stage_name}:")
    logger.info(f"  Average completion: {eval_results['avg_rate']:.1%} (±{eval_results['std_rate']:.1%})")
    logger.info(f"  Min/Max: {eval_results['min_rate']:.1%} / {eval_results['max_rate']:.1%}")
    logger.info(f"  Average reward: {eval_results['avg_reward']:.1f}")
    logger.info(f"  Training time: {training_time/60:.1f} minutes")
    
    # Save model
    save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/models_masked"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{stage_name}_masked.zip")
    model.save(model_path)
    
    # Save results
    results = {
        'stage': stage_name,
        'avg_rate': eval_results['avg_rate'],
        'std_rate': eval_results['std_rate'],
        'min_rate': eval_results['min_rate'],
        'max_rate': eval_results['max_rate'],
        'avg_reward': eval_results['avg_reward'],
        'training_time_min': training_time / 60,
        'timesteps': timesteps,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(save_dir, f"{stage_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return eval_results['avg_rate']


def main():
    """Train all toy stages with action masking."""
    
    # Current best without masking
    current_best = {
        'toy_easy': 1.0,     # Already perfect
        'toy_normal': 0.562, # 56.2%
        'toy_hard': 0.30,    # 30%
        'toy_multi': 0.364   # 36.4%
    }
    
    stages_to_train = ['toy_normal', 'toy_hard', 'toy_multi']
    
    print("\n" + "="*70)
    print("ACTION MASKING TRAINING - TARGETING 80% COMPLETION")
    print("="*70)
    print("\nAction masking filters out invalid actions BEFORE the model sees them.")
    print("This should dramatically improve learning efficiency.")
    print("\nCurrent best (without masking):")
    for stage, rate in current_best.items():
        if stage != 'toy_easy':
            print(f"  {stage}: {rate:.1%}")
    print("\nTarget: 80%+ for all stages")
    print("-"*70)
    
    results = {}
    
    for stage in stages_to_train:
        rate = train_stage_with_masking(stage, timesteps=500000)
        results[stage] = rate
        
        if rate >= 0.8:
            print(f"\n✓ {stage} ACHIEVED 80% TARGET! ({rate:.1%})")
        else:
            print(f"\n✗ {stage} fell short: {rate:.1%} (target: 80%)")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS - ACTION MASKING")
    print("="*70)
    print(f"\n{'Stage':<15} {'Without Masking':<20} {'With Masking':<20} {'Improvement':<15}")
    print("-"*70)
    
    all_achieved = True
    for stage in stages_to_train:
        old_rate = current_best[stage]
        new_rate = results[stage]
        improvement = new_rate - old_rate
        
        print(f"{stage:<15} {old_rate:<20.1%} {new_rate:<20.1%} {improvement:+15.1%}")
        
        if new_rate < 0.8:
            all_achieved = False
    
    if all_achieved:
        print("\n✓ ALL STAGES ACHIEVED 80% TARGET WITH ACTION MASKING!")
        print("Action masking was the key to success!")
    else:
        print("\n✗ Some stages still below 80% even with masking")
        print("May need longer training or different reward structure")


if __name__ == "__main__":
    main()