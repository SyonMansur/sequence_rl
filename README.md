# Benchmarking Biologically Plausible Learning Rules in a Pattern Violation Task

This repository contains code for my **Winter 2026 Rotation Project**, investigating biologically plausible alternatives to backpropagation within a Deep Q-Learning framework. For the analysis, I implement Difference Target Propagation (DTP) and Predictive Coding (PC).

## Task: Gabor Sequence Violation
The agent is trained on a pattern violation task based on Gillon et al. (2024).
* **Architecture:** Feed-forward neural network (FFNN) with two hidden layers.
* **Phase 1 (Habituation):** The agent learns a repeating temporal sequence of $sin/cos$ encoded Gabor patches.
* **Phase 2 (Violations):** 6% of trials involve a pattern violation where the target is shifted by 90°.
* **Action Space:** 16 discrete choices for sequence completion.
* **Reward:** Binary (1 for correct, 0 for miss).

## Learning Rules

### Difference Target Propagation (DTP) (Lee et al. 2015)
DTP replaces global gradients with local **Targets**. It utilizes separate, learned **Inverse Neural Networks** to translate a target from a higher layer back to a lower one. Since inverse networks are often imperfect, a correction term is used to stabilize targets.

### Predictive Coding (PC)
PC minimizes surprise by negotiating common ground between layers. It generates top-down guesses about sensory input and only processes the **prediction error** (mismatch) between those guesses and reality.

Model dynamics are mapped to cortical data from the somata and distal apical dendrites:
* **Model Layer 1** corresponds to **Cortical Layer 2/3**.
* **Model Layer 2** corresponds to **Cortical Layer 5**.
* **Soma Activity:** Modelled by forward pass activations.
* **Apical Dendrite Activity:** Modelled by top-down error signals (backward pass).

Feel free to reach out to syonmansur@g.ucla.edu if you have any questions. Here is a link to my final rotation presentation surrounding these topics: https://docs.google.com/presentation/d/1T1vU_UEvI5vlUEh12o0buR_G0fa_X_g9H3IW2x7_aHM/edit?usp=sharing
