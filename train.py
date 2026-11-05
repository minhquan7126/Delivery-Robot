
from __future__ import annotations
import argparse
import os
import pickle
import numpy as np

from envs.delivery_env import make_env, RewardConfig
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent


def run_episode_qlearning(env, agent, epsilon, use_mask):
    state, info = env.reset()
    mask = info["action_mask"] if use_mask else None
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = agent.select_action(state, epsilon, action_mask=mask)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["action_mask"] if use_mask else None
        agent.update(state, action, reward, next_state, done, next_action_mask=next_mask)
        state = next_state
        mask = next_mask
        total_reward += reward
        steps += 1
    return total_reward, steps, info.get("success", False)


def run_episode_sarsa(env, agent, epsilon, use_mask):
    state, info = env.reset()
    mask = info["action_mask"] if use_mask else None
    action = agent.select_action(state, epsilon, action_mask=mask)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = info["action_mask"] if use_mask else None
        if done:
            next_action = 0  # khong dung den vi done=True trong update
        else:
            next_action = agent.select_action(next_state, epsilon, action_mask=next_mask)
        agent.update(state, action, reward, next_state, next_action, done)
        state, action, mask = next_state, next_action, next_mask
        total_reward += reward
        steps += 1
    return total_reward, steps, info.get("success", False)


def train_one_seed(setting, algo, episodes, seed, invalid_mode, reward_config):
    env = make_env(setting, invalid_action_mode=invalid_mode, reward_config=reward_config)
    use_mask = (invalid_mode == "mask")

    agent_cls = QLearningAgent if algo == "qlearning" else SarsaAgent
    agent = agent_cls(
        n_actions=env.n_actions, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_episodes=int(episodes * 0.7), seed=seed,
    )

    returns = np.zeros(episodes)
    steps_log = np.zeros(episodes)
    success_log = np.zeros(episodes, dtype=bool)

    run_fn = run_episode_qlearning if algo == "qlearning" else run_episode_sarsa

    for ep in range(episodes):
        epsilon = agent.epsilon_at(ep)
        total_reward, steps, success = run_fn(env, agent, epsilon, use_mask)
        returns[ep] = total_reward
        steps_log[ep] = steps
        success_log[ep] = success

    return env, agent, returns, steps_log, success_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--algo", type=str, required=True, choices=["qlearning", "sarsa"])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5, help="so seed chay lien tiep")
    parser.add_argument("--invalid-mode", type=str, default="penalty", choices=["penalty", "mask"])
    parser.add_argument("--reward-config", type=str, default="default",
                         choices=["default", "low_invalid_penalty", "high_step_penalty"])
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.reward_config == "default":
        rc = RewardConfig()
    elif args.reward_config == "low_invalid_penalty":
        rc = RewardConfig(invalid_action_penalty=-2.0)
    elif args.reward_config == "high_step_penalty":
        rc = RewardConfig(step_penalty=-3.0)

    # In thong tin state space
    probe_env = make_env(args.setting, invalid_action_mode=args.invalid_mode, reward_config=rc)
    print(f"=== Setting {args.setting} | algo={args.algo} | invalid_mode={args.invalid_mode} "
          f"| reward_config={args.reward_config} ===")
    print(probe_env.describe_state_space())

    all_returns = []
    all_success = []
    last_env, last_agent = None, None

    for seed in range(args.seeds):
        env, agent, returns, steps_log, success_log = train_one_seed(
            args.setting, args.algo, args.episodes, seed, args.invalid_mode, rc
        )
        all_returns.append(returns)
        all_success.append(success_log)
        print(f"  seed {seed}: mean return (last 200 ep) = {returns[-200:].mean():.2f} | "
              f"success rate (last 200 ep) = {success_log[-200:].mean():.2f} | "
              f"states visited = {agent.n_states_visited()}")
        last_env, last_agent = env, agent

    all_returns = np.array(all_returns)       # shape (seeds, episodes)
    all_success = np.array(all_success)

    tag = f"s{args.setting}_{args.algo}_{args.invalid_mode}_{args.reward_config}"
    out_path = os.path.join(args.outdir, f"{tag}.npz")
    np.savez_compressed(out_path, returns=all_returns, success=all_success)
    print(f"Saved training log -> {out_path}")

    # Luu Q-table cua seed cuoi cung de dung cho evaluate.py / visualize.py
    qtable_path = os.path.join(args.outdir, f"{tag}_qtable.pkl")
    with open(qtable_path, "wb") as f:
        pickle.dump(dict(last_agent.Q), f)
    print(f"Saved Q-table -> {qtable_path}")


if __name__ == "__main__":
    main()
