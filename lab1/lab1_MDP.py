import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import argparse
import sys

# Lớp hỗ trợ ghi log song song ra màn hình và tệp
class Logger(object):
    def __init__(self, filename="lab1_mdp_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class MDPCartPole:
    def __init__(self, env, n_buckets=10, n_samples=10000, progress_every=1000):
        self.env = env
        self.n_buckets = n_buckets
        self.n_samples = n_samples
        self.n_actions = env.action_space.n
        self.progress_every = progress_every
        self.buckets = self._create_buckets()

        # {(state, action): {next_state: (count, total_reward)}}
        self.transitions = defaultdict(
            lambda: defaultdict(lambda: [0, 0.0])
        )

        self.V = defaultdict(float)
        self.policy = defaultdict(int)

        print("Đang ước lượng mô hình từ mẫu thử...")
        self._estimate_model()
    
    def _create_buckets(self):
        bounds = [
            (-4.8, 4.8),        # vị trí xe
            (-3.0, 3.0),        # vận tốc xe
            (-0.418, 0.418),    # góc thanh gỗ
            (-3.0, 3.0)         # vận tốc góc
        ]
        buckets = []
        for low, high in bounds:
            boundaries = np.linspace(low, high, self.n_buckets + 1)[1:-1]
            buckets.append(boundaries)
        return buckets

    def discretize_state(self, state):
        discrete = []
        for i in range(len(state)):
            bucket_index = int(np.digitize(state[i], self.buckets[i]))
            bucket_index = min(max(bucket_index, 0), self.n_buckets - 1)
            discrete.append(bucket_index)
        return tuple(discrete)
    
    def _estimate_model(self):
        for episode in range(self.n_samples):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Phạt nặng nếu ngã để Agent học giữ thăng bằng thay vì chỉ đứng yên
                actual_reward = reward if not terminated else -15.0
                done = terminated or truncated

                s = self.discretize_state(state)
                s2 = self.discretize_state(next_state)

                self.transitions[(s, action)][s2][0] += 1
                self.transitions[(s, action)][s2][1] += actual_reward
                state = next_state

            if self.progress_every and (episode + 1) % self.progress_every == 0:
                print(f"  Đã lấy mẫu {episode + 1}/{self.n_samples} episodes")
        print(f"Số lượng buckets mỗi chiều: {self.n_buckets}")
    
    def get_transition_prob(self, state, action, next_state):
        counts = self.transitions[(state, action)]
        total = sum(c for c, _ in counts.values())
        return counts[next_state][0] / total if total > 0 else 0
    
    def get_expected_reward(self, state, action, next_state):
        count, total_r = self.transitions[(state, action)][next_state]
        return total_r / count if count > 0 else 0

def value_iteration(mdp, gamma=0.99, theta=1e-4, max_iterations=500):
    print("\nĐang chạy VALUE ITERATION...")
    states = list({s for (s, a) in mdp.transitions.keys()})
    iters = 0
    for i in range(max_iterations):
        delta = 0
        V_new = defaultdict(float)
        for s in states:
            action_values = []
            for a in range(mdp.n_actions):
                val = 0
                for s2 in mdp.transitions[(s, a)].keys():
                    p = mdp.get_transition_prob(s, a, s2)
                    r = mdp.get_expected_reward(s, a, s2)
                    val += p * (r + gamma * mdp.V[s2])
                action_values.append(val)
            best_v = max(action_values)
            V_new[s] = best_v
            delta = max(delta, abs(best_v - mdp.V[s]))
        mdp.V = V_new
        if delta < theta:
            iters = i + 1
            break
    if iters == 0: iters = max_iterations

    for s in states:
        best_a, best_val = 0, -float('inf')
        for a in range(mdp.n_actions):
            v = sum(mdp.get_transition_prob(s, a, s2) * (mdp.get_expected_reward(s, a, s2) + gamma * mdp.V[s2])
                    for s2 in mdp.transitions[(s, a)].keys())
            if v > best_val: best_val, best_a = v, a
        mdp.policy[s] = best_a
    return iters

def policy_iteration(mdp, gamma=0.99, max_iterations=100):
    print("\nĐang chạy POLICY ITERATION...")
    states = list({s for (s, a) in mdp.transitions.keys()})
    mdp.V = defaultdict(float)
    for s in states: mdp.policy[s] = random.randint(0, mdp.n_actions - 1)
    
    iters = 0
    for i in range(max_iterations):
        # Policy Evaluation
        for _ in range(30):
            for s in states:
                a = mdp.policy[s]
                mdp.V[s] = sum(mdp.get_transition_prob(s, a, s2) * (mdp.get_expected_reward(s, a, s2) + gamma * mdp.V[s2])
                               for s2 in mdp.transitions[(s, a)].keys())
        # Policy Improvement
        stable = True
        for s in states:
            old_a = mdp.policy[s]
            best_a, max_v = old_a, -float('inf')
            for a in range(mdp.n_actions):
                v = sum(mdp.get_transition_prob(s, a, s2) * (mdp.get_expected_reward(s, a, s2) + gamma * mdp.V[s2])
                        for s2 in mdp.transitions[(s, a)].keys())
                if v > max_v: max_v, best_a = v, a
            mdp.policy[s] = best_a
            if best_a != old_a: stable = False
        if stable:
            iters = i + 1
            print(f"Policy hội tụ tại vòng lặp {iters}")
            break
    if iters == 0: iters = max_iterations
    return iters

def test_policy(env, mdp, episodes=50):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, total_r = False, 0
        while not done:
            s = mdp.discretize_state(state)
            action = mdp.policy.get(s, env.action_space.sample())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += reward
        rewards.append(total_r)
    return np.mean(rewards)

def test_random_agent(env, episodes=100):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, total_r = False, 0
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += reward
        rewards.append(total_r)
    return np.mean(rewards)

if __name__ == "__main__":
    # Thiết lập logging
    sys.stdout = Logger("lab1_mdp_results.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-buckets", type=int, default=10)
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    
    # 1. Khởi tạo MDP
    mdp = MDPCartPole(env, n_buckets=args.n_buckets, n_samples=args.n_samples)

    # 2. Chạy Value Iteration
    it_vi = value_iteration(mdp)
    r_vi = test_policy(env, mdp)

    # 3. Chạy Policy Iteration
    it_pi = policy_iteration(mdp)
    r_pi = test_policy(env, mdp)

    # 4. Chạy Random Agent
    r_rand = test_random_agent(env)

    # 5. In kết quả so sánh
    print("\n--- KẾT QUẢ SO SÁNH ---")
    print(f"Value Iteration:  Reward trung bình = {r_vi:.2f}, Vòng lặp = {it_vi}")
    print(f"Policy Iteration: Reward trung bình = {r_pi:.2f}, Vòng lặp = {it_pi}")
    print(f"Random Agent:     Reward trung bình = {r_rand:.2f}")

    # 6. Vẽ biểu đồ
    labels = ['Value Iteration', 'Policy Iteration', 'Random Agent']
    rewards = [r_vi, r_pi, r_rand]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = plt.bar(labels, rewards, color=colors, edgecolor='black', alpha=0.8)
    
    # Thêm số liệu lên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 3,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.ylabel('Phần thưởng trung bình', fontsize=12)
    plt.title(f'So sánh hiệu quả thuật toán (Samples={args.n_samples}, Buckets={args.n_buckets})', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, max(rewards) + 50)
    
    # Lưu và hiển thị
    plt.savefig("lab1_comparison_plot.png")
    print("\nĐã lưu biểu đồ vào tệp 'lab1_comparison_plot.png'")
    
    # Đóng file log
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal