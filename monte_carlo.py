"""
First-visit Monte Carlo control (epsilon-soft, on-policy) cho Delivery Robot.

Monte Carlo la MODEL-FREE va chi cap nhat Q SAU KHI episode ket thuc:
    G_t = R_{t+1} + gamma * G_{t+1}
    Q(s,a) <- trung binh cua cac G_t tai lan DAU TIEN (s,a) xuat hien trong episode.

Vi du chay:
    python monte_carlo.py --setting 1 --episodes 5000 --seeds 5
    python monte_carlo.py --setting 3 --episodes 30000 --seeds 5
    python monte_carlo.py --setting 2 --episodes 5000 --invalid-mode mask

Dau ra cung dinh dang voi train.py nen summarize.py / visualize.py / evaluate.py dung duoc luon:
    results/s<k>_mc_<invalid-mode>_<reward-config>.npz        (returns, success)
    results/s<k>_mc_<invalid-mode>_<reward-config>_qtable.pkl
"""
from __future__ import annotations
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

from envs.delivery_env import make_env, RewardConfig


class MonteCarloAgent:
    def __init__(self, n_actions, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_episodes=1000, seed=0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start, self.eps_end = epsilon_start, epsilon_end
        self.eps_decay = max(1, epsilon_decay_episodes)
        self.rng = np.random.default_rng(seed)
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
        self.N = defaultdict(lambda: np.zeros(n_actions, dtype=np.int64))

    def epsilon_at(self, ep):
        frac = min(1.0, ep / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, epsilon, action_mask=None):
        valid = np.arange(self.n_actions) if action_mask is None else np.flatnonzero(action_mask)
        if len(valid) == 0:
            valid = np.arange(self.n_actions)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        q = self.Q[state][valid]
        best = valid[np.flatnonzero(q == q.max())]      # random tie-break
        return int(self.rng.choice(best))

    def update_episode(self, trajectory):
        """trajectory = [(state, action, reward), ...]. First-visit MC, trung binh cong don."""
        first_visit = {}
        for t, (s, a, _) in enumerate(trajectory):
            first_visit.setdefault((s, a), t)
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            s, a, r = trajectory[t]
            G = r + self.gamma * G
            if first_visit[(s, a)] == t:
                self.N[s][a] += 1
                self.Q[s][a] += (G - self.Q[s][a]) / self.N[s][a]

    def n_states_visited(self):
        return len(self.Q)


def run_episode(env, agent, epsilon, use_mask, max_steps):
    state, info = env.reset()
    mask = info["action_mask"] if use_mask else None
    trajectory, total_reward, success = [], 0.0, False
    for _ in range(max_steps):
        action = agent.select_action(state, epsilon, action_mask=mask)
        next_state, reward, terminated, truncated, info = env.step(action)
        trajectory.append((state, action, reward))
        total_reward += reward
        state = next_state
        mask = info["action_mask"] if use_mask else None
        if terminated or truncated:
            success = info.get("success", False)
            break
    agent.update_episode(trajectory)
    return total_reward, len(trajectory), success


def make_reward_config(name):
    if name == "low_invalid_penalty":
        return RewardConfig(invalid_action_penalty=-2.0)
    if name == "high_step_penalty":
        return RewardConfig(step_penalty=-3.0)
    return RewardConfig()


def train_one_seed(setting, episodes, seed, invalid_mode, rc, gamma, max_steps):
    env = make_env(setting, invalid_action_mode=invalid_mode, reward_config=rc)
    use_mask = (invalid_mode == "mask")
    agent = MonteCarloAgent(env.n_actions, gamma=gamma,
                            epsilon_decay_episodes=int(episodes * 0.7), seed=seed)
    returns = np.zeros(episodes)
    success = np.zeros(episodes, dtype=bool)
    for ep in range(episodes):
        returns[ep], _, success[ep] = run_episode(env, agent, agent.epsilon_at(ep),
                                                  use_mask, max_steps)
    return env, agent, returns, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=500,
                        help="tran so buoc/episode (setting 1,2 khong co deadline nen can chan tren)")
    parser.add_argument("--invalid-mode", type=str, default="penalty", choices=["penalty", "mask"])
    parser.add_argument("--reward-config", type=str, default="default",
                        choices=["default", "low_invalid_penalty", "high_step_penalty"])
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rc = make_reward_config(args.reward_config)

    probe = make_env(args.setting, invalid_action_mode=args.invalid_mode, reward_config=rc)
    print(f"=== Setting {args.setting} | algo=mc | invalid_mode={args.invalid_mode} "
          f"| reward_config={args.reward_config} ===")
    print(probe.describe_state_space())

    all_returns, all_success, last_agent = [], [], None
    for seed in range(args.seeds):
        env, agent, returns, success = train_one_seed(
            args.setting, args.episodes, seed, args.invalid_mode, rc, args.gamma, args.max_steps)
        all_returns.append(returns)
        all_success.append(success)
        print(f"  seed {seed}: mean return (last 200 ep) = {returns[-200:].mean():.2f} | "
              f"success rate (last 200 ep) = {success[-200:].mean():.2f} | "
              f"states visited = {agent.n_states_visited()}")
        last_agent = agent

    tag = f"s{args.setting}_mc_{args.invalid_mode}_{args.reward_config}"
    out_path = os.path.join(args.outdir, f"{tag}.npz")
    np.savez_compressed(out_path, returns=np.array(all_returns), success=np.array(all_success))
    print(f"Saved training log -> {out_path}")

    qtable_path = os.path.join(args.outdir, f"{tag}_qtable.pkl")
    with open(qtable_path, "wb") as f:
        pickle.dump(dict(last_agent.Q), f)
    print(f"Saved Q-table -> {qtable_path}")


if __name__ == "__main__":
    main()
