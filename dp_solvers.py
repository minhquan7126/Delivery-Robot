
import argparse
import os
import pickle
import time

import numpy as np

from envs.delivery_env import make_env, RewardConfig
from optimal_vi import set_state


def build_model(env):
    """Duyet cac state reachable, tra ve (s0, trans) voi trans[s] = [(ns, r, done) cho moi action]."""
    s0, _ = env.reset()
    trans = {}
    stack = [s0]
    while stack:
        s = stack.pop()
        if s in trans:
            continue
        trans[s] = []
        for a in range(env.n_actions):
            set_state(env, s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            trans[s].append((ns, r, done))
            if not done and ns not in trans:
                stack.append(ns)
    return s0, trans


def q_from_v(trans, V, gamma):
    return {s: np.array([r + (0.0 if done else gamma * V[ns]) for ns, r, done in trans[s]])
            for s in trans}


def value_iteration(trans, gamma, tol=1e-10, max_iter=5000):
    V = {s: 0.0 for s in trans}
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        delta = 0.0
        for s in trans:
            best = max(r + (0.0 if done else gamma * V[ns]) for ns, r, done in trans[s])
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol:
            break
    Q = q_from_v(trans, V, gamma)
    policy = {s: int(np.argmax(Q[s])) for s in trans}
    return V, policy, n_iter


def policy_iteration(trans, gamma, tol=1e-10, max_rounds=1000):
    """Policy Iteration: lap (policy evaluation -> policy improvement) den khi policy on dinh."""
    policy = {s: 0 for s in trans}          # policy ban dau: luon di len
    V = {s: 0.0 for s in trans}
    total_eval_sweeps = 0
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        # --- Policy evaluation (lap den hoi tu) ---
        while True:
            delta = 0.0
            for s in trans:
                ns, r, done = trans[s][policy[s]]
                new_v = r + (0.0 if done else gamma * V[ns])
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            total_eval_sweeps += 1
            if delta < tol:
                break
        # --- Policy improvement (greedy theo V) ---
        stable = True
        for s in trans:
            q = [r + (0.0 if done else gamma * V[ns]) for ns, r, done in trans[s]]
            best = int(np.argmax(q))
            if q[best] > q[policy[s]] + 1e-9:   # nguong de tranh doi policy vi sai so so hoc
                policy[s] = best
                stable = False
        if stable:
            break
    return V, policy, rounds, total_eval_sweeps


def rollout(trans, s0, policy, max_steps=200):
    """Chay policy tat dinh, tra ve (return khong chiet khau, so buoc, ket thuc?)."""
    s, ret, steps, finished = s0, 0.0, 0, False
    while steps < max_steps:
        ns, r, done = trans[s][policy[s]]
        ret += r
        steps += 1
        s = ns
        if done:
            finished = True
            break
    return ret, steps, finished


def make_reward_config(name):
    if name == "low_invalid_penalty":
        return RewardConfig(invalid_action_penalty=-2.0)
    if name == "high_step_penalty":
        return RewardConfig(step_penalty=-3.0)
    return RewardConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="0 = chay ca 4 setting")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-config", type=str, default="default",
                        choices=["default", "low_invalid_penalty", "high_step_penalty"])
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rc = make_reward_config(args.reward_config)
    settings = [1, 2, 3, 4] if args.setting == 0 else [args.setting]

    print("| Setting | |S| reachable | VI: so vong | VI: thoi gian (s) | PI: so round | "
          "PI: tong sweep evaluation | PI: thoi gian (s) | max|V_VI - V_PI| | Return toi uu | So buoc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    for st in settings:
        env = make_env(st, reward_config=rc)
        s0, trans = build_model(env)

        t0 = time.time()
        V_vi, pol_vi, n_vi = value_iteration(trans, args.gamma)
        t_vi = time.time() - t0

        t0 = time.time()
        V_pi, pol_pi, n_rounds, n_sweeps = policy_iteration(trans, args.gamma)
        t_pi = time.time() - t0

        diff = max(abs(V_vi[s] - V_pi[s]) for s in trans)
        ret_vi, steps_vi, _ = rollout(trans, s0, pol_vi)
        ret_pi, steps_pi, _ = rollout(trans, s0, pol_pi)
        note = "" if abs(ret_vi - ret_pi) < 1e-9 else f" (PI: {ret_pi:.1f}/{steps_pi})"

        print(f"| {st} | {len(trans)} | {n_vi} | {t_vi:.2f} | {n_rounds} | {n_sweeps} | "
              f"{t_pi:.2f} | {diff:.2e} | {ret_vi:.1f}{note} | {steps_vi} |")

        for tag, V in [("vi", V_vi), ("pi", V_pi)]:
            path = os.path.join(args.outdir, f"s{st}_{tag}_penalty_{args.reward_config}_qtable.pkl")
            with open(path, "wb") as f:
                pickle.dump(q_from_v(trans, V, args.gamma), f)


if __name__ == "__main__":
    main()
