"""Adjusted hyperparameters targeting 80% - final attempt"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class SimpleRewardWrapper(Monitor):
    """Dead simple rewards - just schedule jobs"""
    def __init__(self, env):
        super().__init__(env)
        self.scheduled = 0
        
    def reset(self, **kwargs):
        self.scheduled = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        
        # Simple reward
        if info.get('action_valid', False) and info.get('action_type') == 'schedule':
            reward = 100.0
            self.scheduled += 1
        else:
            reward = -1.0
            
        # Completion bonus
        if done or truncated:
            total = self.env.total_tasks if hasattr(self.env, 'total_tasks') else 1
            rate = self.scheduled / total
            reward += 1000.0 * rate  # Proportional to completion
            
        return obs, reward, done, truncated, info


def train_with_adjusted_params(stage_name):
    print(f"\nTraining {stage_name} with adjusted hyperparameters...")
    
    env = SimpleRewardWrapper(CurriculumEnvironmentTrulyFixed(stage_name, verbose=False))
    env = DummyVecEnv([lambda: env])
    
    # ADJUSTED HYPERPARAMETERS
    model = PPO(
        'MlpPolicy',
        env,
        # LEARNING RATE - try different values
        learning_rate=3e-4,  # Lower than before (was 1e-3)
        
        # BATCH SIZE - smaller for more frequent updates  
        n_steps=512,         # Smaller batches (was 2048)
        batch_size=64,       # Smaller batch (was 256)
        
        # TRAINING INTENSITY
        n_epochs=20,         # More epochs per update (was 10)
        
        # EXPLORATION - CRITICAL FOR FINDING VALID ACTIONS
        ent_coef=0.1,        # Balanced exploration (was 0.01-0.5)
        
        # NETWORK SIZE - bigger for complex patterns
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])  # Deeper
        ),
        
        # OTHER
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train for 1M steps
    print("Training for 1M timesteps with adjusted params...")
    model.learn(total_timesteps=1000000)
    
    # Evaluate
    print("\nEvaluating...")
    eval_env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    total_scheduled = 0
    total_possible = 0
    
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
        total_scheduled += scheduled
        total_possible += total
        
    final_rate = total_scheduled / total_possible
    print(f"\nFinal performance: {final_rate:.1%}")
    
    if final_rate >= 0.8:
        save_dir = "/Users/carrickcheah/Project/ppo/app_2/phase3/adjusted_models"
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f"{stage_name}_adjusted.zip"))
        print(f"SUCCESS! Model saved.")
    
    return final_rate


def main():
    print("ADJUSTED HYPERPARAMETERS ATTEMPT")
    print("=" * 50)
    print("Key adjustments:")
    print("- Lower learning rate: 3e-4 (was 1e-3)")
    print("- Smaller batches: 512 steps (was 2048)")
    print("- More epochs: 20 (was 10)")
    print("- Deeper network: 3 layers")
    print("- Balanced exploration: 0.1")
    print("- Longer training: 1M steps")
    print("=" * 50)
    
    results = {}
    for stage in ['toy_normal', 'toy_hard', 'toy_multi']:
        rate = train_with_adjusted_params(stage)
        results[stage] = rate
        
        if rate < 0.8:
            print(f"\nStill below 80%. Possible reasons:")
            print("1. Action space too sparse (~10% valid actions)")
            print("2. Need action masking to guide exploration")
            print("3. Problem may need hierarchical approach")
            print("4. Consider using the 100% sequences we found as demos")
    
    print("\nFINAL RESULTS:")
    for stage, rate in results.items():
        print(f"{stage}: {rate:.1%}")


if __name__ == "__main__":
    main()