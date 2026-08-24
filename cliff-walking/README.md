# Cliff Walking: Q-Learning vs SARSA (Gymnasium)

A Reinforcement Learning comparison studying the behavioral differences between **Off-Policy (Q-Learning)** and **On-Policy (SARSA)** temporal-difference control algorithms in the Gymnasium `CliffWalking-v1` gridworld environment.

---

## 📌 Environment & Problem Statement
* **Environment**: `CliffWalking-v1` (4x12 grid = 48 states, 4 discrete actions: Up, Down, Right, Left).
* **Objective**: Navigate an agent from start `[3, 0]` (state 36) to the goal `[3, 11]` (state 47) without falling off the cliff `[3, 1..10]`.
* **Rewards**: -1 per step, -100 penalty for falling off the cliff.

---

## ⚙️ Algorithms & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate ($\alpha$)** | 0.5 | Step size for TD error updates |
| **Discount Factor ($\gamma$)** | 0.99 | Importance of future rewards |
| **Exploration Rate ($\epsilon$)** | 0.1 | Epsilon-greedy exploration probability |
| **Training Episodes** | 500 | Total learning iterations |

### Algorithm Update Rules

* **Q-Learning (Off-Policy TD Control)**:
  Updates state-action values using the maximum estimated value of the next state regardless of the action taken:
  $$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \right]$$

* **SARSA (On-Policy TD Control)**:
  Updates state-action values using the actual next action chosen by the $\epsilon$-greedy policy:
  $$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

---

## 📊 Key Insights

* **Q-Learning** learns the optimal, shortest path right along the edge of the cliff, but risks falling during exploration due to $\epsilon$-greedy actions.
* **SARSA** learns a safer, longer path farther away from the edge to account for exploratory actions causing cliff drops.

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Q-Learning**:
   ```bash
   python q_learning.py
   ```

3. **Run SARSA**:
   ```bash
   python sarsa.py
   ```