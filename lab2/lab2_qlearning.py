import gymnasium as gym
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import logging
import sys
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
    qtable_txt_file = f"{log_dir}/qtable_{timestamp}.txt"

    # Co gang dua console ve UTF-8 de in ky tu mui ten tren Windows.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("==== FrozenLake Q-Learning Training ====")

    return log_dir, timestamp, plot_file, qtable_file, qtable_txt_file

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
def train(env, Q, episodes, max_steps, alpha, gamma, epsilon, epsilon_decay, epsilon_min, log_every=500):
    rewards = []
    win_flags = []
    wins = 0
    best_avg_reward = -1.0
    best_episode = 0
    best_qtable = None

    for episode in range(episodes):

        state, _ = env.reset()
        total_reward = 0

        episode_win = 0
        for step in range(max_steps):
            action = choose_action(state , Q,   epsilon, env.action_space.n)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if next_state == state:
                reward -= 0.01
            done = terminated or truncated
            update_qtable(Q, state, action, reward, next_state, alpha, gamma)

            state = next_state
            total_reward += reward

            if reward == 1:
                episode_win = 1
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)
        win_flags.append(episode_win)
        wins += episode_win

        if (episode + 1) % log_every == 0:
            avg_reward = np.mean(rewards[-log_every:])
            avg_winrate = np.mean(win_flags[-log_every:])
            logging.info(f"Epsilon {epsilon} | Avg reward: {avg_reward:.3f} | Epsilon: {epsilon:.3f}")
            logging.info(f"Win rate (last {log_every}): {avg_winrate:.3f}")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_episode = episode + 1
                best_qtable = Q.copy()

    logging.info(f"Training wins: {wins}/{episodes}")
    logging.info(f"Best avg reward: {best_avg_reward:.3f} at episode {best_episode}")
    return rewards, win_flags, wins, best_avg_reward, best_episode, best_qtable

# Plot learning curve
def plotting_learningcurve(rewards, win_flags, log_dir, timestamp, window=100):
    file_path = f"{log_dir}/learning_curve_{timestamp}.png"
    episodes = np.arange(1, len(rewards) + 1)
    rewards_arr = np.array(rewards, dtype=float)
    wins_arr = np.array(win_flags, dtype=float)

    # Rolling metrics de giam nhieu va de nhin xu huong ro hon
    roll_window = max(5, window)
    if len(rewards_arr) >= roll_window:
        reward_roll = np.convolve(rewards_arr, np.ones(roll_window) / roll_window, mode="valid")
        win_roll = np.convolve(wins_arr, np.ones(roll_window) / roll_window, mode="valid")
        roll_episodes = np.arange(roll_window, len(rewards_arr) + 1)
    else:
        reward_roll = rewards_arr
        win_roll = wins_arr
        roll_episodes = episodes

    cumulative_winrate = np.cumsum(wins_arr) / episodes

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(episodes, rewards_arr, color="#90caf9", alpha=0.4, linewidth=1, label="Reward (raw)")
    axes[0].plot(roll_episodes, reward_roll, color="#1565c0", linewidth=2, label=f"Reward MA({roll_window})")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Q-learning Training")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(episodes, wins_arr, color="#a5d6a7", alpha=0.4, linewidth=1, label="Win (0/1)")
    axes[1].plot(roll_episodes, win_roll, color="#2e7d32", linewidth=2, label=f"Win rate MA({roll_window})")
    axes[1].plot(episodes, cumulative_winrate, color="#ef6c00", linewidth=1.6, label="Cumulative win rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
    plt.close()

    logging.info(f"Learning curve saved: {file_path}")

def plotting_winrate_curve(win_flags, log_dir, timestamp, window=100):
    file_path = f"{log_dir}/winrate_curve_{timestamp}.png"
    wins = np.array(win_flags, dtype=float)
    block_winrates = []
    block_labels = []

    for start in range(0, len(wins), window):
        end = min(start + window, len(wins))
        block = wins[start:end]
        block_winrates.append(float(np.mean(block)))
        block_labels.append(f"{start + 1}-{end}")

    x = np.arange(len(block_winrates))

    plt.figure(figsize=(10, 6))
    plt.bar(x, block_winrates, color="#4caf50", alpha=0.85, width=0.8)

    if len(block_winrates) <= 20:
        plt.xticks(x, block_labels, rotation=45, ha="right")
    else:
        tick_step = max(1, len(block_winrates) // 10)
        tick_idx = np.arange(0, len(block_winrates), tick_step)
        plt.xticks(tick_idx, [block_labels[i] for i in tick_idx], rotation=45, ha="right")

    plt.xlabel(f"Episode blocks ({window} episodes/block)")
    plt.ylabel("Win rate")
    plt.title("Win Rate (Bar Chart)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    logging.info(f"Win rate curve saved: {file_path}")

# Evualate agent
def evualate_agent(env, Q, tests=100):
    wins = 0

    for _ in range(tests):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward == 1:
                wins += 1

    win_rate = wins / tests

    logging.info(f"Win rate: {win_rate:.2f}")

# Print Q-table after training (formatted table)
def print_qtable(Q, env, qtable_txt_file=None):
    action_symbols = ["←", "↓", "→", "↑"]
    action_symbols_ascii = ["LEFT", "DOWN", "RIGHT", "UP"]

    desc = env.unwrapped.desc
    flat_tiles = []
    for cell in desc.flatten():
        if isinstance(cell, (bytes, bytearray)):
            flat_tiles.append(cell.decode("utf-8"))
        else:
            flat_tiles.append(str(cell))

    lines = []
    lines.append("=" * 60)
    lines.append("Q-TABLE SAU KHI HUAN LUYEN")
    lines.append("=" * 60)
    lines.append(f"{'State':<6}{'←':>10}{'↓':>10}{'→':>10}{'↑':>10}{'Best Action':>14}")
    lines.append("-" * 60)

    for state, row in enumerate(Q):
        tile = flat_tiles[state]
        if tile == "H":
            best_action_text = "[H]"
        elif tile == "G":
            best_action_text = "[G]"
        else:
            best_action_text = action_symbols[int(np.argmax(row))]

        lines.append(
            f"S{state:<5}{row[0]:>10.4f}{row[1]:>10.4f}{row[2]:>10.4f}{row[3]:>10.4f}{best_action_text:>14}"
        )

    lines.append("=" * 60)
    lines.append("Chu thich: H=Hole (terminal), G=Goal (terminal)")
    qtable_str = "\n".join(lines)
    qtable_log = qtable_str
    for uni, asc in zip(action_symbols, action_symbols_ascii):
        qtable_log = qtable_log.replace(uni, asc)

    # Console logger luon dung ASCII de tranh loi cp1252 tren Windows terminal.
    logging.info("\n%s", qtable_log)
    if qtable_txt_file:
        with open(qtable_txt_file, "w", encoding="utf-8") as f:
            f.write(qtable_str + "\n")
        logging.info(f"Q-table saved: {qtable_txt_file}")

# Print best actions (greedy policy) from Q-table
def print_best_actions(Q):
    best_actions = np.argmax(Q, axis=1)
    logging.info(f"Best action per state: {best_actions.tolist()}")
    logging.info(f"Best action at start state (0): {int(best_actions[0])}")


# Main
def main():
    log_dir, timestamp, plot_file, qtable_file, qtable_txt_file = setup_logging()

    env, state_size, action_size = create_env()

    Q = initialize_qtable(state_size, action_size)

    # Hyperameters
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    episodes = 20000
    max_steps = 100

    rewards, win_flags, _, best_avg_reward, best_episode, best_qtable = train(
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

    logging.info(f"Best episode (avg reward): {best_episode}")

    # Use best_qtable if available, otherwise final Q
    final_q = best_qtable if best_qtable is not None else Q
    print_qtable(final_q, env, qtable_txt_file=qtable_txt_file)
    print_best_actions(final_q)

    plotting_learningcurve(rewards, win_flags, log_dir, timestamp)
    plotting_winrate_curve(win_flags, log_dir, timestamp)

    evualate_agent(env, final_q)

    env.close()


if __name__ == "__main__":
    main()




