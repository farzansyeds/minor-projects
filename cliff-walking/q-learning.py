import random
import gymnasium as gym
import numpy as np

# Hyperparameters
gamma = 0.99
epsilon = 0.1
alpha = 0.5
episodes = 500

# Initialize Q-table
Q = np.zeros((48, 4))


# Epsilon-greedy policy
def epsilon_greedy(env, state):
    if random.random() < epsilon:
        return env.action_space.sample()  # explore
    else:
        return np.argmax(Q[state])  # exploit


# Q-learning training loop
for episode in range(episodes):
    # render = episode % 50 == 0

    # if render:
    #     env = gym.make("CliffWalking-v1", render_mode="human")
    # else:
    #     env = gym.make("CliffWalking-v1")

    env = gym.make("CliffWalking-v1")
    

    done = False
    state, _ = env.reset()

    episode_len = 0
    tot_reward = 0

    while not done:
        action = epsilon_greedy(env, state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        episode_len += 1
        tot_reward += reward

    print(
        f"episode={episode+1}/500 & total reward = {tot_reward} & episode len = {episode_len}"
    )
    env.close()

# Evaluate the learned policy
env = gym.make("CliffWalking-v1", render_mode="human")
state, _ = env.reset()
total_reward = 0
episode_len = 0
done = False

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    episode_len += 1
    total_reward += reward

print(f"total reward = {total_reward} & epsiode len = {episode_len}")
env.close()

# Inspect Q-values
print("Q[36]:", Q[36])
print("Q[35]:", Q[35])