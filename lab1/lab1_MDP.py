import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import argparse

class MDPCartPole:
    def __init__(self, env, n_buckets=10, n_samples=10000, progress_every=1000):
        self.env = env
        # n_buckets: so bin moi chieu; n_samples: so episode de lay mau
        self.n_buckets = n_buckets
        self.n_samples = n_samples
        self.n_actions = env.action_space.n
        # in tien do uoc luong model moi progress_every episode
        self.progress_every = progress_every

        # Tao buckets de discretize state
        self.buckets = self._create_buckets()

        # Transition model (dem so lan chuyen tiep + tong reward)
        #{(state, action): {next_state: (count, total_reward)}}
        self.transitions = defaultdict(
            lambda: defaultdict(lambda: [0,0.0])
        )

        # value function va policy
        self.V = defaultdict(float)
        self.policy = defaultdict(int)

        # estimate model from samples
        print("Estimate model from sample...")
        self._estimate_model()
    
    # Create buckets for discretization
    def _create_buckets(self):
        bounds = [
            (-4.8, 4.8),        # cart position
            (-3.0, 3.0),        # cart velocity
            (-0.418, 0.418),    # pole angle
            (-3.0, 3.0)         # pole velocity
        ]
        buckets = []
        for low, high in bounds:
            # Use internal boundaries so digitize returns 0..n_buckets-1
            boundaries = np.linspace(low, high, self.n_buckets + 1)[1:-1]
            buckets.append(boundaries)
        return buckets

    # Convert continuous state -> discrete (chi so bin 0..n_buckets-1)
    def discretize_state(self, state):
        discrete = []

        for i in range(len(state)):
            bucket_index = int(np.digitize(state[i], self.buckets[i]))
            # clamp ve dung mien 0..n_buckets-1
            bucket_index = min(max(bucket_index, 0), self.n_buckets - 1)

            discrete.append(bucket_index)
        return tuple(discrete)
    
    # Estimate transition model tu mau ngau nhien
    def _estimate_model(self):
        for episode in range(self.n_samples):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                s = self.discretize_state(state)
                s2 = self.discretize_state(next_state)

                # dem tan suat chuyen tiep va tong reward de uoc luong P, R
                count, total_r = self.transitions[(s, action)][s2]

                self.transitions[(s, action)][s2][0] = count + 1
                self.transitions[(s, action)][s2][1] = total_r + reward

                state = next_state
            if self.progress_every and (episode + 1) % self.progress_every == 0:
                print(f"  Estimated {episode + 1}/{self.n_samples} episodes")
    
    # Get all next state
    def get_all_next_states(self, state, action):
        return list(self.transitions[(state, action)].keys())
    
    # Transition probability: P(s'|s,a) = count / total_count
    def get_transition_prob(self, state, action, next_state):
        counts = self.transitions[(state, action)]
        total = sum(c for c, _ in counts.values())

        if total == 0 :
            return 0
        return counts[next_state][0] / total
    
    # Expected reward: R_hat(s,a,s') = total_reward / count
    def get_expected_reward(self, state, action, next_state):
        count, total_r = self.transitions[(state, action)][next_state]
        if count == 0:
            return 0
        return total_r / count
    
def value_iteration(mdp, gamma=0.99, theta=1e-6, max_interations=500):
        """
        Value iteration Algorithm

        Args:
            mdp: MDP object
            gamma: Discount factor
            theta: Convergence threshold
            max_iteration: Maximum interations

        Returns:
            V: Optimal value function
            poilicy : optimal policy
            iteration: Number of iteration
        """
        print("\nRunning VALUE ITERATION")

        states = list({s for (s,a) in mdp.transitions.keys()})
        iterations_used = 0
        # Bellman optimality update:
        # V(s) = max_a sum_{s'} P(s'|s,a) * (R + gamma * V(s'))
        for interation in range(max_interations):
            delta = 0
            V_new = defaultdict(float)

            for s in states:
                action_values = []

                for a in range(mdp.n_actions):
                    value = 0
                    next_states = mdp.get_all_next_states(s, a)

                    for s2 in next_states:
                        p = mdp.get_transition_prob(s, a, s2)
                        r = mdp.get_expected_reward(s, a, s2)

                        value += p * (r + gamma * mdp.V[s2])

                    action_values.append(value)
                best_value = max(action_values)
                V_new[s] = best_value
                # delta theo doi muc thay doi lon nhat
                delta = max(delta, abs(best_value - mdp.V[s]))
            mdp.V = V_new

            if delta < theta:
                iterations_used = interation + 1
                print("Converged after", iterations_used, "iterations")
                break
        if iterations_used == 0:
            iterations_used = max_interations
        
        # extract policy tu V da hoi tu
        for s in states:
            best_action = 0
            best_value = -1e9
            for a in range(mdp.n_actions):
                value = 0
                for s2 in mdp.get_all_next_states(s, a):
                    p = mdp.get_transition_prob(s, a, s2)
                    r = mdp.get_expected_reward(s, a, s2)

                    value += p * (r + gamma *mdp.V[s2])
                
                if value > best_value:
                    best_value = value
                    best_action = a
            mdp.policy[s] = best_action
        return defaultdict(float, mdp.V), defaultdict(int, mdp.policy), iterations_used
    
# POLICY ITERATION
def policy_iteration(mdp, gamma=0.99, max_iterations=100):

    print("\nRunning POLICY ITERATION")

    states = list({s for (s, a) in mdp.transitions.keys()})
    # reset value function de so sanh cong bang voi Value Iteration
    mdp.V = defaultdict(float)

    # initialize random policy
    for s in states:
        mdp.policy[s] = random.randint(0, mdp.n_actions - 1)

    iterations_used = 0
    for iteration in range(max_iterations):

        # Policy Evaluation (lap co dinh 50 vong de xap xi V_pi)
        for _ in range(50):

            for s in states:

                a = mdp.policy[s]

                value = 0

                for s2 in mdp.get_all_next_states(s, a):

                    p = mdp.get_transition_prob(s, a, s2)
                    r = mdp.get_expected_reward(s, a, s2)

                    value += p * (r + gamma * mdp.V[s2])

                mdp.V[s] = value

        # Policy Improvement: chon action toi uu theo V hien tai
        stable = True

        for s in states:

            old_action = mdp.policy[s]

            best_action = old_action
            best_value = -1e9

            for a in range(mdp.n_actions):

                value = 0

                for s2 in mdp.get_all_next_states(s, a):

                    p = mdp.get_transition_prob(s, a, s2)
                    r = mdp.get_expected_reward(s, a, s2)

                    value += p * (r + gamma * mdp.V[s2])

                if value > best_value:

                    best_value = value
                    best_action = a

            mdp.policy[s] = best_action

            if best_action != old_action:
                stable = False

        if stable:
            iterations_used = iteration + 1
            print("Policy converged at iteration", iterations_used)
            break
    if iterations_used == 0:
        iterations_used = max_iterations

    return defaultdict(float, mdp.V), defaultdict(int, mdp.policy), iterations_used


# TEST POLICY

def test_policy(env, mdp, episodes=20):

    rewards = []

    for _ in range(episodes):

        state, _ = env.reset()

        done = False
        total_reward = 0

        while not done:

            s = mdp.discretize_state(state)

            # neu state chua gap trong model thi dung action ngau nhien
            action = mdp.policy.get(s, env.action_space.sample())

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    avg_reward = np.mean(rewards)

    print("\nAverage reward:", avg_reward)

    return avg_reward

# RANDOM AGENT BASELINE
def test_random_agent(env, episodes=100):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        # baseline: chon action ngau nhien moi buoc
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return float(np.mean(rewards))
# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDP + DP for CartPole")
    # tham so co the thay doi khi chay de thu nghiem
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    env = gym.make("CartPole-v1")

    log_dir = "lab1/logs/lab1_MDP"
    os.makedirs(log_dir, exist_ok=True)

    results_path = f"{log_dir}/lab1_mdp_results.txt"

    with open(results_path, "w") as log_file:

        mdp = MDPCartPole(
            env,
            n_buckets=args.n_buckets,
            n_samples=args.n_samples,
            progress_every=args.progress_every,
        )

        # VALUE ITERATION
        V_vi, policy_vi, vi_iters = value_iteration(mdp)
        reward_vi = test_policy(env, mdp, episodes=args.test_episodes)

        # POLICY ITERATION
        V_pi, policy_pi, pi_iters = policy_iteration(mdp)
        reward_pi = test_policy(env, mdp, episodes=args.test_episodes)

        # RANDOM AGENT BASELINE
        reward_random = test_random_agent(env, episodes=args.random_episodes)

        print("\nComparison:")
        print("Value Iteration reward:", reward_vi)
        print("Policy Iteration reward:", reward_pi)
        print("Random Agent reward:", reward_random)

        # Model statistics: quy mo quan sat thuc te tu mau
        states = {s for (s, a) in mdp.transitions.keys()}
        num_states = len(states)
        num_state_actions = len(mdp.transitions)
        num_transitions = sum(len(nxt) for nxt in mdp.transitions.values())

        log_file.write("Model estimation episodes: " + str(mdp.n_samples) + "\n")
        log_file.write("Discrete buckets per dimension: " + str(mdp.n_buckets) + "\n")
        log_file.write("Observed discrete states: " + str(num_states) + "\n")
        log_file.write("Observed state-action pairs: " + str(num_state_actions) + "\n")
        log_file.write("Observed transitions: " + str(num_transitions) + "\n\n")

        log_file.write("Value Iteration iterations: " + str(vi_iters) + "\n")
        log_file.write("Value Iteration reward: " + str(reward_vi) + "\n")
        log_file.write("Policy Iteration iterations: " + str(pi_iters) + "\n")
        log_file.write("Policy Iteration reward: " + str(reward_pi) + "\n")
        log_file.write("Random Agent reward: " + str(reward_random) + "\n")




