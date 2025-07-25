"""
Quick test to verify Small Rush rewards are fixed
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from phase3.improve_stage_performance import StageImprover


def quick_test():
    """Quick test of Small Rush stage."""
    print("=== QUICK SMALL RUSH TEST ===\n")
    
    # Create stage improver
    improver = StageImprover('small_rush')
    
    # Create environment
    print("Creating environment...")
    env_fn = improver.create_env
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Create model with high entropy
    print("\nCreating PPO model...")
    print(f"- Entropy coefficient: {improver.stage_config.get('ent_coef', 0.05)}")
    print(f"- Reward profile: {improver.stage_config.get('reward_profile', 'balanced')}")
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=2e-4,
        n_steps=256,  # Small for quick test
        batch_size=64,
        n_epochs=5,
        ent_coef=improver.stage_config.get('ent_coef', 0.05),
        verbose=1
    )
    
    # Train for a short time
    print("\nTraining for 5000 steps...")
    model.learn(total_timesteps=5000)
    
    # Test the model
    print("\n=== TESTING TRAINED MODEL ===")
    obs = vec_env.reset()
    total_reward = 0
    actions_taken = 0
    
    for i in range(100):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        
        if reward[0] > 0:
            actions_taken += 1
            total_reward += reward[0]
        
        if done[0]:
            obs = vec_env.reset()
    
    print(f"\nResults after 100 steps:")
    print(f"- Actions with positive reward: {actions_taken}")
    print(f"- Total reward: {total_reward:.2f}")
    print(f"- Average reward: {total_reward/max(actions_taken, 1):.2f}")
    
    if actions_taken > 0:
        print("\n✅ SUCCESS: Model is taking actions and getting rewards!")
        print("The 0% utilization fix is working.")
    else:
        print("\n❌ FAILURE: Model still not taking actions.")
    
    return actions_taken > 0


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n\n🎉 Small Rush fix confirmed! Ready to run full training.")
        print("\nNext step: cd phase3 && python improve_stage_performance.py small_rush")
    else:
        print("\n\n⚠️ Fix not working. Check reward calculations.")