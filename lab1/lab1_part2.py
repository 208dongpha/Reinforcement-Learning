import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)  # In ra console
        self.log.write(message)       # Ghi vào file

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Explore environment
def explore_env(env_name, log_file=None):
    env = gym.make(env_name)

    msgs = []
    msgs.append(f"Environment: {env_name}")
    msgs.append(f"Observation space: {env.observation_space}")
    msgs.append(f"Action space: {env.action_space}")

    obs, info = env.reset()
    msgs.append(f"Initial observation: {obs}")
    msgs.append(f"Observation dimension: {len(obs)}")

    for msg in msgs:
        print(msg)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    env.close()

def random_agent(env_name, episodes=100, log_file=None):
    env = gym.make(env_name)
    all_rewards = []
    
    # Giúp in các mảng Numpy(ví dụ: 4 chữ số thập phân)
    np.set_printoptions(precision=4, suppress=True)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\n{'='*20} START EPISODE {ep+1} ({env_name}) {'='*20}")

        while not done:
            # Chọn hành động ngẫu nhiên
            action = env.action_space.sample()
            
            # In State hiện tại và Action được chọn
            # State: trạng thái agent nhìn thấy trước khi hành động
            # Action: hành động agent thực hiện tại trạng thái đó
            print(f"Step {step_count:3d} | State: {obs} | Action: {action}")
            
            # Thực thi hành động
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        all_rewards.append(total_reward)
        print(f"--- Episode {ep+1} Finished | Total Steps: {step_count} | Total Reward: {total_reward} ---")

    avg_reward = np.mean(all_rewards)
    print(f"\n{env_name} Average Reward over {episodes} episodes: {avg_reward:.2f}")

    env.close()
    return all_rewards

# Moving average
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Vẽ learning curve
def plot_learning_curve(rewards, env_name, log_dir, window=10):
    episodes = len(rewards)
    mov_avg = moving_average(rewards, window)
    avg_reward = np.mean(rewards)

    plt.figure(figsize=(12,6))
    plt.plot(range(1, episodes+1), rewards, label="Reward per Episode", alpha=0.6)
    plt.plot(range(window, episodes+1), mov_avg, label=f"Moving Average (window={window})", color='orange', linewidth=2)
    plt.axhline(y=avg_reward, color='red', linestyle='--', label=f"Average Reward ({avg_reward:.2f})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Random Agent Learning Curve - {env_name}")
    plt.legend()

    image_path = os.path.join(log_dir, f"{env_name}_learning_curve.png")
    plt.savefig(image_path)
    print(f"Saved learning curve image: {image_path}")
    plt.close()

if __name__ == "__main__":
    log_dir = "lab1/logs/lab1_part2"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "episode_log.txt")

    open(log_file, "w").close()

    # Redirect tất cả print vào file log
    sys.stdout = Logger(log_file)

    episodes = 100
    window = 10

    # Explore environments
    explore_env("CartPole-v1")
    explore_env("MountainCar-v0")

    # CartPole
    cartpole_rewards = random_agent("CartPole-v1", episodes)
    np.save(os.path.join(log_dir, "cartpole_reward.npy"), cartpole_rewards)
    print(f"CartPole average reward: {np.mean(cartpole_rewards):.2f}")
    plot_learning_curve(cartpole_rewards, "CartPole-v1", log_dir, window)

    # MountainCar
    mountain_rewards = random_agent("MountainCar-v0", episodes)
    np.save(os.path.join(log_dir, "mountain_reward.npy"), mountain_rewards)
    print(f"MountainCar average reward: {np.mean(mountain_rewards):.2f}")
    plot_learning_curve(mountain_rewards, "MountainCar-v0", log_dir, window)

    print(f"\nAll results (rewards, images, episode log) saved in {log_dir}")