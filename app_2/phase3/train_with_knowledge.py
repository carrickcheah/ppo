"""Train with knowledge that 100% IS possible - use curriculum learning"""

import os
import sys
import json
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class CurriculumWrapper(Monitor):
    """Curriculum learning - start with huge rewards, gradually make realistic"""
    
    def __init__(self, env, stage_name):
        super().__init__(env)
        self.stage_name = stage_name
        self.episode_count = 0
        self.scheduled_count = 0
        self.curriculum_phase = 0  # 0: Pure completion, 1: Balanced, 2: Realistic
        
    def reset(self, **kwargs):
        self.scheduled_count = 0
        self.episode_count += 1
        
        # Progress curriculum every 100 episodes
        if self.episode_count % 100 == 0 and self.curriculum_phase < 2:
            self.curriculum_phase += 1
            print(f"[{self.stage_name}] Curriculum phase: {self.curriculum_phase}")
            
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, original_reward, done, truncated, info = super().step(action)
        
        # Phase 0: Pure completion focus
        if self.curriculum_phase == 0:
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                reward = 1000.0
                self.scheduled_count += 1
            elif info.get('action_valid', False):
                reward = 1.0
            else:
                reward = -0.1
                
        # Phase 1: Balanced
        elif self.curriculum_phase == 1:
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                reward = 500.0
                self.scheduled_count += 1
            elif info.get('action_valid', False):
                reward = 0.5
            else:
                reward = -1.0
                
        # Phase 2: More realistic but still completion-focused
        else:
            if info.get('action_valid', False) and info.get('action_type') == 'schedule':
                reward = 100.0 + max(0, original_reward)  # Add original if positive
                self.scheduled_count += 1
            elif info.get('action_valid', False):
                reward = 0.1
            else:
                reward = -5.0
        
        # Always give completion bonus
        if done or truncated:
            total = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            rate = self.scheduled_count / total if total > 0 else 0
            
            if rate >= 1.0:
                reward += 5000.0 * (3 - self.curriculum_phase)  # Bigger bonus in early phases
            elif rate >= 0.9:
                reward += 2000.0 * (3 - self.curriculum_phase)
            elif rate >= 0.8:
                reward += 1000.0 * (3 - self.curriculum_phase)
        
        return obs, reward, done, truncated, info


def train_with_curriculum(stage_name, expected_rate=1.0):
    """Train using curriculum learning"""
    print(f"\nTraining {stage_name} (expected: {expected_rate:.1%})...")
    
    # Create environment
    base_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    env = CurriculumWrapper(base_env, stage_name)
    env = DummyVecEnv([lambda: env])
    
    # Model with balanced hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.2,  # Good exploration
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=0
    )
    
    # Train for 300k timesteps
    print("Training with curriculum learning...")
    start_time = time.time()
    
    # Checkpoint callback
    save_dir = f"/Users/carrickcheah/Project/ppo/app_2/phase3/curriculum_models/{stage_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="checkpoint"
    )
    
    model.learn(total_timesteps=300000, callback=checkpoint_callback, progress_bar=True)
    
    # Evaluate final performance
    print("\nEvaluating final model...")
    eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    results = []
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = eval_env.step(action)
            done = done or truncated
            steps += 1
        
        scheduled = len(eval_env.scheduled_jobs) if hasattr(eval_env, 'scheduled_jobs') else 0
        total = eval_env.total_tasks if hasattr(eval_env, 'total_tasks') else 1
        rate = scheduled / total
        results.append(rate)
        
        if ep == 0:
            print(f"First episode: {scheduled}/{total} = {rate:.1%}")
    
    final_rate = np.mean(results)
    std_rate = np.std(results)
    
    print(f"Final performance: {final_rate:.1%} (±{std_rate:.1%})")
    
    # Save model and results
    model_path = os.path.join(save_dir, "final_model.zip")
    model.save(model_path)
    
    results_data = {
        'stage': stage_name,
        'final_rate': final_rate,
        'std_rate': std_rate,
        'expected_rate': expected_rate,
        'training_time': time.time() - start_time,
        'model_path': model_path
    }
    
    with open(os.path.join(save_dir, "results.json"), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return final_rate


def main():
    print("CURRICULUM LEARNING FOR 100% COMPLETION")
    print("=" * 60)
    print("We KNOW 100% is possible from analysis!")
    print("Using curriculum learning to achieve it.")
    print("=" * 60)
    
    # Expected rates from analysis
    stages = {
        'toy_normal': 1.0,  # 100% is possible
        'toy_hard': 1.0,    # 100% is possible
        'toy_multi': 0.955  # 95.5% is possible
    }
    
    results = {'toy_easy': 1.0}  # Already perfect
    
    for stage, expected in stages.items():
        rate = train_with_curriculum(stage, expected)
        results[stage] = rate
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS WITH CURRICULUM LEARNING")
    print("="*60)
    
    all_achieved = True
    for stage in ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']:
        if stage in results:
            rate = results[stage]
            expected = stages.get(stage, 1.0)
            
            if stage == 'toy_easy':
                status = "✓ Already perfect"
            elif rate >= expected - 0.01:  # Within 1% of expected
                status = f"✓ {rate:.1%} (target: {expected:.1%})"
            else:
                status = f"✗ {rate:.1%} (target: {expected:.1%})"
                all_achieved = False
                
            print(f"{stage}: {status}")
    
    if all_achieved:
        print("\n✓ ALL TOYS ACHIEVED TARGET PERFORMANCE!")
        print("Ready to proceed to the next phase!")
    else:
        print("\n✗ Some stages need more work")


if __name__ == "__main__":
    main()