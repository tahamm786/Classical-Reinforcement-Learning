#  Classical Reinforcement Learning 
A collection of foundational RL implementations completed while preparing for a larger DRL-based autonomous driving project.

Before moving into vision-based control and training an autonomous vehicle using Deep Reinforcement Learning, I spent time building a strong understanding of fundamental RL concepts. This repository contains those learning-phase implementations, small, focused assignments that helped me internalize how value functions, policies, and neural networks work at a low level.

These projects were intentionally created from scratch to strengthen intuition without relying heavily on pre-built RL libraries.

---

##  What’s Included

### FrozenLake: Value-Based Methods
A classic tabular environment used to understand how agents reason about states and actions.

Implemented:
- **Policy Iteration**
- **Value Iteration**

Key learnings:
- How value functions are computed through iterative Bellman backups  
- The relationship between policy evaluation and policy improvement  
- How dynamic programming methods converge  
- The contrast between model-based DP and model-free RL approaches  

---

### MiniGrid : Monte Carlo & Temporal-Difference Learning
MiniGrid offers a more challenging setting with partial observability and sparse rewards. It served as a great environment to experiment with model-free control methods.

Implemented:
- **Monte Carlo (Every-Visit)**
- **Monte Carlo with Constant Step Size (α)**
- **Q-Learning**
- **SARSA**

Key learnings:
- Designing and tuning ε-greedy exploration  
- Differences between MC, TD(0), on-policy, and off-policy updates  
- Handling stochasticity and delayed rewards  
- Practical considerations for stability and convergence  

---

### Neural Network From Scratch (XOR)
A minimal neural network built manually to solve the classic XOR problem.

Covered:
- Forward propagation  
- Backpropagation  
- Gradient descent updates  

Why this mattered:
- XOR is a simple but powerful example of non-linearity  
- Building the network by hand clarified how gradients move through layers  
- Provided a strong foundation before working with CNNs in the main autonomous driving project  

---

##  Purpose of This Repository  
These assignments represent the learning steps that led up to my main DRL project, where I trained an autonomous car using the DonkeyCar simulator and PPO from Stable-Baselines3. The goal of this repository is to document those early explorations and the conceptual foundations that prepared me for working with a full control pipeline, continuous actions, and vision-based inputs provided by the simulator.

---

##  Skills Strengthened
- Working with MDPs and Bellman equations  
- Implementing Monte Carlo and TD learning from first principles  
- Understanding on-policy vs. off-policy control  
- Exploring strategies for exploration and reward structuring  
- Building and training neural networks manually  
- Developing intuition useful for scaling up to CNN-based DRL  

---

## 📌 Related Main Project (Separate Repository)  
These assignments supported a larger project where I built a DRL agent that drives autonomously in the DonkeyCar simulator using raw pixel observations.

