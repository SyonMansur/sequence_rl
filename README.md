# Deep Q Reinforcement Learning and Third Factors

This repository contains code for my **Winter 2026 Rotation Project**, investigating sensorimotor learning through the lens of deep reinforcement learning.

### Deep Q-Learning & Dopamine
The project utilizes a **Deep Q-Network (DQN)** to map sensory inputs (sine/cosine pairs) to a high-resolution action space. **Mean Squared Error (MSE)** loss serves as an artificial proxy for **Reward Prediction Error (RPE)**.
The model mimics the phasic firing of midbrain dopamine neurons by minimizing the difference between predicted value ($Q$) and received reward ($R$).

### Third Factors in Plasticity
This project explores **third factors** in synaptic plasticity. Specifically, I am investigating how global signals facilitate learning during rule-shift phases and how top-down gradients modify early sensory layers (`fc1`) responsible for late rewards (credit assignment).

This project is currently a work in progress (updated 1/26/26)
