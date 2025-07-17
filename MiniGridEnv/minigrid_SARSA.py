import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
import time
import matplotlib.pyplot as plt
import wandb

wandb.init(project="Q_LEARNING", name="q-run", config={
    "episodes": 1000,
    "alpha": 0.1,
    "gamma": 0.95
})



env = gym.make("MiniGrid-Empty-6x6-v0",render_mode=None)

valid_actions=[0,1,2]
n_actions = len(valid_actions)

Q = defaultdict(lambda: np.zeros(n_actions))
E = defaultdict(lambda: np.zeros(n_actions))

def get_state(env):
    
    raw_env=env.unwrapped
    return (raw_env.agent_pos[0], raw_env.agent_pos[1], raw_env.agent_dir)

def epsilon_greedy(state, Q, epsilon, n_actions):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        return np.argmax(Q[state])


def sarsa_lambda_train(env, Q, E, episodes, alpha, gamma, lam, n_actions):
    rewards = []
    steps = []
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.995

    

    for ep in range(episodes):
        env.reset()
        state = get_state(env)
        action = epsilon_greedy(state, Q, epsilon, n_actions)

        
        for key in E:
            E[key].fill(0.0)

        total_reward, count, done = 0.0, 0, False

        while not done:
            obs, reward, term, trunc, _ = env.step(action)
            next_state = get_state(env)
            next_action = epsilon_greedy(next_state, Q, epsilon, n_actions)

            td_target = reward + gamma * Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1

            for s in Q:
                Q[s] += alpha * td_target * E[s]
                E[s] *= gamma * lam

            state = next_state
            action = next_action
            total_reward += reward
            count += 1
            done = term or trunc

        rewards.append(total_reward)
        steps.append(count)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        wandb.log({

            "Episode": ep,
            "Reward": total_reward,
            "Steps": count,
            "Epsilon": epsilon

        })

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Steps: {count} | epsilon: {epsilon:.3f}")

    return Q, rewards, steps


Q, rewards, steps = sarsa_lambda_train(env,Q,E,episodes=500,alpha=0.1,gamma=0.95,lam=0.9,n_actions=n_actions)


def simple_moving_average(data, window_size=100):
    averaged = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        averaged.append(np.mean(data[start:i+1]))
    return averaged

smoothed_rewards = simple_moving_average(rewards, window_size=100)
smoothed_steps = simple_moving_average(steps, window_size=100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(smoothed_rewards, label='Average Reward',color="orange")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Reward per Episode - SARSA LAMBDA")

plt.subplot(1, 2, 2)
plt.plot(smoothed_steps, label='Average Steps', color='orange')
plt.xlabel("Episode")
plt.ylabel("Average Steps")
plt.title("Steps per Episode - SARSA LAMBDA")

plt.tight_layout()
plt.show()

for ep in range(len(smoothed_rewards)):
    wandb.log({
        "Smoothed Reward": smoothed_rewards[ep],
        "Smoothed Steps": smoothed_steps[ep],
        "Episode (smoothed)": ep
    })

policy = {}

for state in Q:
    best_action = np.argmax(Q[state])
    policy[state] = best_action


env.close()


test_env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="human")
test_env.reset()
state = get_state(test_env)

done = False
total_reward = 0
steps=0

while not done:
    action = policy.get(state, random.choice(valid_actions))
    obs, reward, term, trunc, _ = test_env.step(action)
    time.sleep(1.0)  
    total_reward += reward
    state = get_state(test_env)
    done = term or trunc
    steps+=1

print("Test episode reward:", total_reward)
print("Test episode steps:", steps)
test_env.close()
