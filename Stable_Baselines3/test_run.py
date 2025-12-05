import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1",render_mode='human')

# Create agent
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="log")

# Train agent
model.learn(total_timesteps=10_000)

# Test agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

