import gymnasium as gym
import numpy as np
import time

# Environment
env = gym.make("FrozenLake-v1",render_mode="human", is_slippery=True)
env = env.unwrapped  # Get full access to P table

num_states = env.observation_space.n
num_actions = env.action_space.n

# Initialize a random policy (1 action per state)
policy = np.random.choice(num_actions, size=num_states)
# Value function
V = np.zeros(num_states)
gamma = 0.99

is_policy_stable = False
iteration = 0

while not is_policy_stable:
    iteration += 1
    print(f"\n Iteration {iteration}...")

    # ---- Step 1: Policy Evaluation ----
    epsilon = 1e-4
    while True:
        delta = 0
        for s in range(num_states):
            v = 0
            a = policy[s]
            for prob, next_state, reward, done in env.P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < epsilon:
            break

    # ---- Step 2: Policy Improvement ----
    is_policy_stable = True
    for s in range(num_states):
        old_action = policy[s]
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(action_values)
        policy[s] = best_action
        if old_action != best_action:
            is_policy_stable = False


arrow_map = np.array(['←', '↓', '→', '↑'])
print("Current Policy:")
print(arrow_map[policy].reshape(4, 4)) 

def run_policy(env, policy, max_steps=100):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = policy[obs]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        time.sleep(0.8)  # Pause to see the agent moving

        if terminated or truncated:
            print(f"\nEpisode finished after {step + 1} steps.")
            print(f"Total reward: {total_reward}")
            break
    else:
        print(" Episode Terminated - No. of steps exceeded 100")
        print(f" Total reward: {total_reward}")


run_policy(env,policy)

