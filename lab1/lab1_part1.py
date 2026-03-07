import gymnasium as gym
import os

def analyze_env(env_name):
    env = gym.make(env_name)
    obs, info = env.reset()

    result = []
    result.append(f"===== {env_name} =====")
    result.append(f"Observation space: {env.observation_space}")
    result.append(f"Action space: {env.action_space}")

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    result.append(f"Sample reward: {reward}")
    result.append(f"Terminated: {terminated}")
    result.append(f"Truncated: {truncated}")
    result.append("")

    env.close()
    return "\n".join(result)

if __name__ == "__main__":
    log_dir = "lab1/logs/lab1_part1"
    os.makedirs(log_dir, exist_ok=True)

    cartpole_info = analyze_env("CartPole-v1")
    mountaincar_info = analyze_env("MountainCar-v0")


    # lưu vào file
    with open(os.path.join(log_dir, "env_analysis.txt"), "w") as f:
        f.write(cartpole_info)
        f.write("\n")
        f.write(mountaincar_info)

    print("Saved environment analysis to logs/lab1_part1/env_analysis.txt")

