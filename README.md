This project focuses on building an autonomous vehicle that can navigate a track by following lanes using Deep Reinforcement Learning (DRL). The agent learns to control steering and throttle purely from raw camera images using Convolutional Neural Networks (CNNs) and RL techniques.

🔧 Environment
Simulator: gym-donkeycar

Actions:
    Steering: Continuous range from -5 to +5
    Throttle: Continuous range from 0 to 1

Observations: Raw pixel images from a front-facing camera

Rewards: Based on lane-centering and maintaining speed

Termination: Episode ends if the car veers off track or crashes

🧠 What I Learned
Applied Deep Reinforcement Learning to a real-world-inspired control problem.<br>
Processed visual input through CNNs for policy learning.<br>
Tuned and trained agents using reward shaping and environment feedback.<br>
Developed a robust control policy capable of handling dynamic driving conditions.<br>
Gained experience in training, evaluating, and optimizing AI models for real-time decision making.<br>

🏁 Goal

Build a self-driving agent that:

Stays centered in the lane<br>
Maintains optimal speed<br>
Navigates a full lap successfully without collisions<br>