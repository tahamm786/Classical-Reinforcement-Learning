import gymnasium as gym
import time

# Create the environment with GUI rendering
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)

# Number of episodes to run
num_episodes = 3

for episode in range(1, num_episodes + 1):
    print(f"\n=== Starting Episode {episode} ===")
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
    
    step_count = 0

    while True:
        action = env.action_space.sample()  # Random action
        new_obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Print step info
        print(f"Step {step_count}:")
        print(f"  Action taken      : {action}")
        print(f"  New observation   : {new_obs}")
        print(f"  Reward            : {reward}")
        print(f"  Terminated        : {terminated}")
        print(f"  Truncated         : {truncated}")
        print(f"  Info              : {info}")
        print("-" * 40)

        time.sleep(1)  # Delay to visualize step

        obs = new_obs

        if terminated or truncated:
            result = "Reached Goal!" if reward == 1.0 else "Fell in Hole!"
            print(f"\nEpisode {episode} finished after {step_count} steps: {result}")
            break

    time.sleep(2)  # Pause before starting next episode

# Close the environment window after all episodes
env.close()
