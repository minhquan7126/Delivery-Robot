"""
So sanh 5 thuat toan (Value Iteration, Policy Iteration, Q-learning, SARSA, Monte Carlo)
tren CUNG mot thuoc do: chat luong policy cuoi cung, do tren mo hinh P, R cua moi truong.

Chay (sau khi da train xong Q-learning, SARSA, Monte Carlo cho setting do):
    python compare_all.py --setting 1
    python compare_all.py --setting 2 --resultsdir results

Can san cac file (do train.py va monte_carlo.py tao ra, che do penalty, reward default):
    results/s<k>_qlearning_penalty_default_qtable.pkl   va  .npz
    results/s<k>_sarsa_penalty_default_qtable.pkl       va  .npz
    results/s<k>_mc_penalty_default_qtable.pkl          va  .npz

Cac cot cua bang:
  - Return rollout: chay policy tham lam (khong ngau nhien) tu o Start, cong reward KHONG chiet khau.
  - Gap V(s0): v*(s0) - v_pi(s0) (co chiet khau, tinh tren mo hinh). 0 = toi uu tai state dau.
  - % state toi uu: ti le state reachable ma hanh dong cua policy nam trong tap hanh dong toi uu.
  - % state da tham: ti le state reachable ma agent tung ghe tham (Q-table co khoa do); VI/PI = 100%.
  - Chi phi: DP -> k, so sweep, thoi gian; RL -> so episode de return TB (cua so 100) dat >= 0.
LUU Y: don vi chi phi cua DP (sweep) va RL (episode) KHONG so sanh truc tiep duoc voi nhau.
Q-table chi la cua seed cuoi cung (xem train.py), nen so lieu chat luong policy la cua 1 seed.
"""
import argparse
import os
import pickle
import time

import numpy as np

from envs.delivery_env import make_env
from dp_solvers import build_model, make_reward_config
import dp_analysis as dpa

RL_ALGOS = [("qlearning", "Q-learning"), ("sarsa", "SARSA"), ("mc", "Monte Carlo")]


def policy_from_q(m, qdict):
    """Policy tham lam (argmax Q) cho moi state reachable; state chua tham -> action 0."""
    pol = np.zeros(m.n, dtype=np.int64)
    visited = 0
    for i, s in enumerate(m.states):
        q = qdict.get(s)
        if q is None:
            continue
        visited += 1
        pol[i] = int(np.argmax(q))
    return pol, visited / m.n


def rollout(m, i0, pol, max_steps=200):
    i, ret = i0, 0.0
    for t in range(1, max_steps + 1):
        a = pol[i]
        ret += m.R[i, a]
        if m.D[i, a]:
            return ret, t, True
        i = m.NS[i, a]
    return ret, max_steps, False


def first_reach(values, window, thr):
    if len(values) < window:
        return None
    ma = np.convolve(values, np.ones(window) / window, mode="valid")
    idx = np.where(ma >= thr)[0]
    return int(idx[0] + window) if len(idx) else None


def metrics(m, i0, pol, Vstar, qstar):
    V, _ = dpa.evaluate(m, pol, np.zeros(m.n), None)
    gap = Vstar[i0] - V[i0]
    rows = np.arange(m.n)
    pct_opt = float(np.mean(qstar[rows, pol] >= qstar.max(axis=1) - 1e-6))
    ret, steps, ok = rollout(m, i0, pol)
    return ret, steps, ok, gap, pct_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--reward-config", type=str, default="default",
                        choices=["default", "low_invalid_penalty", "high_step_penalty"])
    args = parser.parse_args()

    env = make_env(args.setting, reward_config=make_reward_config(args.reward_config))
    s0, trans = build_model(env)
    m = dpa.Model(trans, args.gamma)
    i0 = m.idx[s0]
    Vstar = dpa.solve_vstar(m)
    qstar = m.q(Vstar)
    opt_ret, opt_steps, _ = rollout(m, i0, qstar.argmax(axis=1))

    rows = []

    # ---- Value Iteration
    t0 = time.time()
    vi = dpa.truncated_pi(m, Vstar, 1, 5000)
    t_vi = time.time() - t0
    pol_vi = qstar.argmax(axis=1)
    rows.append(("Value Iteration", "model-based", pol_vi, 1.0,
                 f"k={len(vi['errs']) - 1}, {vi['sweeps']} sweep, {t_vi:.2f}s"))

    # ---- Policy Iteration
    t0 = time.time()
    pi = dpa.truncated_pi(m, Vstar, None, 5000)
    pol_pi = dpa.policy_iteration_history(m)[-1][0]
    t_pi = time.time() - t0
    rows.append(("Policy Iteration", "model-based", pol_pi, 1.0,
                 f"k={len(pi['errs']) - 1}, {pi['sweeps']} sweep, {t_pi:.2f}s"))

    # ---- Q-learning, SARSA, Monte Carlo
    tag = f"_penalty_{args.reward_config}"
    for key, name in RL_ALGOS:
        qpath = os.path.join(args.resultsdir, f"s{args.setting}_{key}{tag}_qtable.pkl")
        npath = os.path.join(args.resultsdir, f"s{args.setting}_{key}{tag}.npz")
        if not os.path.exists(qpath):
            print(f"[bo qua] khong thay {qpath}")
            continue
        with open(qpath, "rb") as f:
            qdict = pickle.load(f)
        pol, frac = policy_from_q(m, qdict)
        cost = "(khong co file .npz)"
        if os.path.exists(npath):
            data = np.load(npath)
            n_ep = data["returns"].shape[1]
            reach = first_reach(data["returns"].mean(axis=0), 100, 0.0)
            cost = (f"return>=0 sau {reach} ep (tong {n_ep} ep)" if reach is not None
                    else f"chua dat return>=0 (tong {n_ep} ep)")
        rows.append((name, "model-free", pol, frac, cost))

    print(f"\nSetting {args.setting} | gamma={args.gamma} | |S| reachable={m.n} | "
          f"return toi uu tu Start = {opt_ret:.1f} ({opt_steps} action)\n")
    print("| Thuat toan | Nhom | Return rollout | So action | Gap V(s0) | % state toi uu | % state da tham | Chi phi hoc |")
    print("|---|---|---|---|---|---|---|---|")
    for name, grp, pol, frac, cost in rows:
        ret, steps, ok, gap, pct = metrics(m, i0, pol, Vstar, qstar)
        ret_s = f"{ret:.1f}" if ok else "khong giao duoc"
        print(f"| {name} | {grp} | {ret_s} | {steps if ok else '-'} | {gap:.2f} | {100 * pct:.1f}% | "
              f"{100 * frac:.0f}% | {cost} |")


if __name__ == "__main__":
    main()
