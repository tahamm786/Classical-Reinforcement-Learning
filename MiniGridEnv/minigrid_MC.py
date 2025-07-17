import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
import time
import matplotlib.pyplot as plt
import wandb

wandb.init(project="MC", name="mc-run", config={
    "episodes": 1000,
    "alpha": 0.1,
    "gamma": 0.99
})

env = gym.make("MiniGrid-Empty-6x6-v0",render_mode=None)

valid_actions=[0,1,2]
n_actions=len(valid_actions)

Q = defaultdict(lambda: np.zeros(len(valid_actions))) 



def get_state(env):
    
    #(x, y, dir) taken straight from the env.
   
    raw_env=env.unwrapped
    return (raw_env.agent_pos[0], raw_env.agent_pos[1], raw_env.agent_dir)


def epsilon_greedy(state, Q, epsilon, n_actions):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        return np.argmax(Q[state])


def generate_episode(env, policy, epsilon):
    episode = []
    env.reset()
    state = get_state(env)
    done = False
    
    while not done:
        # ε-greedy action selection
        action=epsilon_greedy(state,Q,epsilon,n_actions)
        _, reward, terminated, truncated, _ = env.step(action)
        next_state = get_state(env)
        done = terminated or truncated

        episode.append((state, action, reward))
        state = next_state
    

    return episode


def every_visit_mc_control(env, episodes, gamma, epsilon,epsilon_min=0.001, epsilon_decay=0.99):
    returns = defaultdict(list)  # to store all returns for each state, action
     
    policy = dict()  # the current policy 

    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(episodes):
        episode = generate_episode(env, policy, epsilon)
        
        total_reward = sum(r for _, _, r in episode)
        rewards_per_episode.append(total_reward)

        steps_per_episode.append(len(episode))


        #for all steps in episode - every visit / for single visit u will need to do the update only once for each state-action pair
        for i, (state, action, _) in enumerate(episode):
            # Compute return G from time step i to end
            G = 0
            for j in range(i, len(episode)):
                _, _, reward = episode[j]
                G += (gamma ** (j - i)) * reward


            
            returns[(state, action)].append(G)
            Q[state][action] = np.mean(returns[(state, action)])

            # Improve policy: make it ε-greedy w.r.t. Q
            best_action = np.argmax(Q[state])
            probs = np.ones(len(valid_actions)) * (epsilon / len(valid_actions))
            probs[best_action] += (1 - epsilon)
            policy[state] = np.random.choice(valid_actions, p=probs)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        wandb.log({

            "Episode": ep,
            "Reward": total_reward,
            "Steps": len(episode),
            "Epsilon": epsilon

        })

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{episodes} | Reward : {total_reward :.2f} | Steps : {len(episode)} | epsilon : {epsilon:.3f} ")

    return Q, policy , rewards_per_episode, steps_per_episode

Q, policy , rewards, steps = every_visit_mc_control(env, episodes=1000, gamma=0.99, epsilon=1.0)

def simple_moving_average(data, window_size=100):
    averaged = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        averaged.append(np.mean(data[start:i+1]))
    return averaged


smoothed_rewards = simple_moving_average(rewards, window_size=100)
smoothed_steps = simple_moving_average(steps, window_size=100)

# Reward per episode
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.plot(smoothed_rewards)
plt.title("Total Reward per Episode - MONTE CARLO")
plt.xlabel("Episode")
plt.ylabel("Reward")

# Steps per episode
plt.subplot(1, 2, 2)
plt.plot(steps)
plt.plot(smoothed_steps)
plt.title("Steps per Episode - MONTE CARLO")
plt.xlabel("Episode")
plt.ylabel("Number of Steps")

plt.tight_layout()
plt.show()

for ep in range(len(smoothed_rewards)):
    wandb.log({
        "Smoothed Reward": smoothed_rewards[ep],
        "Smoothed Steps": smoothed_steps[ep],
        "Episode (smoothed)": ep
    })



test_env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="human", max_steps=100)

total_reward = 0
test_env.reset()
s = get_state(test_env)
done = False
step=0
while not done:

    test_env.render() 
    if s in policy:
        a = policy[s]
    else:
        a = random.choice(valid_actions)

    _, r, term, trunc, _ = test_env.step(a)

          
    time.sleep(0.5) 

    total_reward += r
    s = get_state(test_env)
    step+=1
    done = term or trunc

test_env.render()
time.sleep(1.0)
test_env.close()
print(f"Episode return:, {total_reward:.2f}")
print("No. of Steps",step)

wandb.finish()