import gymnasium as gym
import numpy as np
import time

# Environment
env = gym.make("FrozenLake-v1",render_mode='human', map_name="4x4", is_slippery=True)
env = env.unwrapped

# Number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n

# Discount factor
gamma = 0.99

V = np.zeros(num_states)

epsilon = 1e-4  # for error
max_iterations = 1000  # to prevent infinite loops
iteration = 0

while True:
    delta = 0  # tracks maximum change in value
    new_V = np.zeros(num_states)

    for s in range(num_states):
        q_sa = []
        for a in range(num_actions):
            transitions = env.P[s][a]
            value = 0
            for prob, next_state, reward, done in transitions:
                value += prob * (reward + gamma * V[next_state])
            q_sa.append(value)
        
        best_action_value = max(q_sa)
        delta = max(delta, abs(best_action_value - V[s]))
        new_V[s] = best_action_value

    V = new_V
    iteration += 1

    if delta < epsilon or iteration >= max_iterations:
        break

print(f"Value iteration converged in {iteration} iterations.")

policy = np.zeros(num_states, dtype=int)

for s in range(num_states):
    q_sa = []
    for a in range(num_actions):
        value = 0
        for prob, next_state, reward, done in env.P[s][a]:
            value += prob * (reward + gamma * V[next_state])
        q_sa.append(value)
    
    best_action = np.argmax(q_sa)
    policy[s] = best_action





def run_policy(env, policy, max_steps=100):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        time.sleep(0.6)  #to slow 

        action = policy[obs]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\n Episode finished after {step + 1} steps.")
            print(f" Total reward: {total_reward}")
            break
    else:
        print("\n  Episode did not finish within max steps.")


run_policy(env, policy)


