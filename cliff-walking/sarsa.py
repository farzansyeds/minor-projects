import random
import gymnasium as gym
import numpy as np

# Sample environment inspection
env = gym.make("CliffWalking-v1")
print("States:", env.observation_space.n)
print("Actions:", env.action_space.n)

starting_state, _ = env.reset()
print("Starting state:", starting_state)
env.close()

# Hyperparameters
gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

# Initializing Q-table
Q = np.zeros((48, 4))


# Epsilon-greedy policy
def epsilon_greedy(env, state):
    if random.random() < epsilon:
        return env.action_space.sample()  # random action => EXPLORE
    else:
        return np.argmax(Q[state])  # exploit


# SARSA training loop
for episode in range(episodes):
    # render = episode % 50 == 0

    # if render:
    #     env = gym.make("CliffWalking-v1", render_mode="human")
    #     print(f"rendering env for episode={episode+1}/500")
    # else:
    #     env = gym.make("CliffWalking-v1")

    env= gym.make("CliffWalking-v1")

    done = False
    state, _ = env.reset()
    action = epsilon_greedy(env, state)

    total_reward = 0
    episode_len = 0

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy(env, next_state)

        # SARSA update
        Q[state, action] += alpha * (
            reward + gamma * (Q[next_state, next_action] - Q[state, action])
        )

        state = next_state
        action = next_action

        total_reward += reward
        episode_len += 1

    print(
        f"episode={episode+1}/500: total reward = {total_reward} & ep length = {episode_len}"
    )
    env.close()

# Evaluate the learned policy
env = gym.make("CliffWalking-v1", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0
episode_len = 0

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward
    episode_len += 1

print(f"total reward = {total_reward} & episode len = {episode_len}")
env.close()