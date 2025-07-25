"""
Quick test to train on Toy Easy stage
"""

import os
import sys
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.curriculum_env import CurriculumSchedulingEnv


def test_toy_easy_training():
    """Quick training test on Toy Easy."""
    print("=== TRAINING TOY EASY TEST ===")
    
    # Stage config
    stage_config = {
        'name': 'toy_easy',
        'jobs': 5,
        'machines': 3,
        'description': 'Learn sequence rules',
        'multi_machine_ratio': 0.0
    }
    
    # Create environment
    def make_env():
        env = CurriculumSchedulingEnv(
            stage_config=stage_config,
            data_source="synthetic",
            reward_profile="learning",
            seed=42
        )
        env = Monitor(env)
        return env
    
    # Vectorized environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create simple PPO model (using MlpPolicy for quick test)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,  # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )
    
    print("\nTraining for 10,000 timesteps...")
    model.learn(total_timesteps=10000)
    
    # Test the trained model
    print("\n=== TESTING TRAINED MODEL ===")
    test_env = CurriculumSchedulingEnv(
        stage_config=stage_config,
        data_source="synthetic",
        reward_profile="learning",
        seed=123
    )
    
    obs, info = test_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    # Results
    test_env._calculate_final_metrics()
    metrics = test_env.episode_metrics
    
    print(f"\nResults:")
    print(f"  Jobs completed: {metrics['jobs_completed']}/{test_env.n_jobs}")
    print(f"  Machine utilization: {metrics['machine_utilization']:.1%}")
    print(f"  Jobs late: {metrics['jobs_late']}")
    print(f"  Total reward: {total_reward:.2f}")
    
    # Save model if good
    if metrics['machine_utilization'] > 0.8:
        os.makedirs("checkpoints/toy_easy", exist_ok=True)
        model.save("checkpoints/toy_easy/test_model")
        env.save("checkpoints/toy_easy/test_vecnorm.pkl")
        print(f"\n✓ Model saved! Achieved {metrics['machine_utilization']:.1%} utilization")
    else:
        print(f"\n✗ Model needs more training. Only {metrics['machine_utilization']:.1%} utilization")


if __name__ == "__main__":
    test_toy_easy_training()