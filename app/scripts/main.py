# Copy this EXACTLY and run it!
import gymnasium as gym
from stable_baselines3 import PPO

# 1. Make a simple game environment
env = gym.make("CartPole-v1")

# 2. Create AI brain
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the AI (let it practice)
print("Training started... wait 30 seconds!")
model.learn(total_timesteps=10_000)
print("Training done!")

# 4. Watch the trained AI play!
obs, _ = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

        