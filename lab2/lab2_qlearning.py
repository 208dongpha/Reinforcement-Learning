import gymnasium as gym
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import logging
from datetime import datetime

# Setup logging
def setup_logging():

    # Create log folder
    log_dir = "lab2/logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = f"{log_dir}/training_{timestamp}.log"
    plot_file = f"{log_dir}/learning_curve_{timestamp}.png"
    qtable_file = f"{log_dir}/qtable_{timestamp}.npy"

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("==== FrozenLake Q-Learning Training ====")

    return log_dir, timestamp, plot_file, qtable_file

# Initialize Env
def create_env():

    env = gym.make("FrozenLake-v1", is_slippery=True)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    logging.info(f"Number of states: {state_size}")
    logging.info(f"Number of actions: {action_size}")
    return env, state_size, action_size

# Initialize Qtable
def initialize_qtable(state_size, action_size):
    Q = np.zeros((state_size, action_size))

    logging.info("Initialize Q-table: ")
    logging.info(Q)
    return Q

# Choose action (epsilon greedy)
def choose_action(state, Q, epsilon, action_size):
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, action_size - 1)
    else:
        action = np.argmax(Q[state])
    return action

# Q-learning update rule
def update_qtable(Q, state, action, reward, next_state, alpha, gamma):
    old_value = Q[state, action]

    Q[state, action] = old_value + alpha * (
        reward + gamma * np.max(Q[next_state]) - old_value
    )

# Tranning loop
def train(env, Q, episodes, max_steps, alpha, gamma, epsilon, epsilon_decay, epsilon_min, ):
    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = choose_action(state , Q,   epsilon, env.action_space.n)
            next_state, reward, done, truncated, _ = env.step(action)
            update_qtable(Q, state, action, reward, next_state, alpha, gamma)

            state = next_state
            total_reward += reward

            if done:    
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)

        if episode % 500 == 0:
            avg_reward = np.mean(rewards[-500:])
            logging.info(f"Epsilon {epsilon} | Avg reward: {avg_reward:.3f} | Epsilon: {epsilon:.3f}")

    return rewards

# Plot learning curve
def plotting_learningcurve(rewards, log_dir, timestamp):
    file_path = f"{log_dir}/learning_curve_{timestamp}.png"

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning Training")

    plt.savefig(file_path)

    logging.info(f"Learning curve saved: {file_path}")

# Evualate agent
def evualate_agent(env, Q, tests=100):
    wins = 0

    for _ in range(tests):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, done, truncated, _ = env.step(action)

            if reward == 1:
                wins += 1

    win_rate = wins / tests

    logging.info(f"Win rate: {win_rate:.2f}")

# Main
def main():
    log_dir, timestamp, plot_file, qtable_file = setup_logging()

    env, state_size, action_size = create_env()

    Q = initialize_qtable(state_size, action_size)

    # Hyperameters
    alpha = 0.8
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    episodes = 5000
    max_steps = 100

    rewards = train(
        env,
        Q, 
        episodes,
        max_steps,
        alpha,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min
    ) 

    plotting_learningcurve(rewards, log_dir, timestamp)

    evualate_agent(env, Q)

    env.close()


if __name__ == "__main__":
    main()




