"""
Ve learning curve (tu file .npz do train.py / monte_carlo.py xuat ra) va policy arrow-grid
(tu file Q-table .pkl) cho tung setting/carrying value.

BAN MO RONG so voi ban dau:
  - learning-curve ve toi da 3 duong (--npz, --npz2, --npz3)
  - --hline VALUE : ve duong ngang (vi du return toi uu 36 hoac 41)
  - --band        : to bong vung min-max giua cac seed (de thay do dao dong)
  - --metric success : ve success rate thay vi return (dung cho Fig. 7, setting 3 va 4)
Lenh `policy` giu nguyen.

Vi du chay:
    # Fig. 4: Q-learning, SARSA, Monte Carlo + duong toi uu (setting 2 -> 36)
    python visualize.py learning-curve --npz results/s2_qlearning_penalty_default.npz \
        --npz2 results/s2_sarsa_penalty_default.npz --npz3 results/s2_mc_penalty_default.npz \
        --label1 "Q-learning" --label2 "SARSA" --label3 "Monte Carlo" \
        --hline 36 --band --out figs/s2_fig4.png

    # Fig. 6: penalty so voi mask (hai duong)
    python visualize.py learning-curve --npz results/s2_qlearning_penalty_default.npz \
        --npz2 results/s2_qlearning_mask_default.npz --label1 "Penalty" --label2 "Mask" \
        --out figs/s2_compare_invalid_mode.png

    # Fig. 7: success rate theo episode (setting 3)
    python visualize.py learning-curve --metric success --npz results/s3_qlearning_penalty_default.npz \
        --npz2 results/s3_sarsa_penalty_default.npz --npz3 results/s3_mc_penalty_default.npz \
        --label1 "Q-learning" --label2 "SARSA" --label3 "Monte Carlo" --out figs/s3_success.png

    python visualize.py policy --setting 1 --qtable results/s1_qlearning_penalty_default_qtable.pkl \
        --invalid-mode penalty --out figs/s1_policy.png
"""
from __future__ import annotations
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from envs.delivery_env import make_env, ACTIONS

ARROW = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}  # Up,Down,Left,Right (dx,dy for quiver-like text)


def moving_average(x, window=100):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def ensure_dir(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_learning_curve(args):
    fig, ax = plt.subplots(figsize=(8, 5))

    runs = [(args.npz, args.label1)]
    if args.npz2:
        runs.append((args.npz2, args.label2))
    if args.npz3:
        runs.append((args.npz3, args.label3))

    for path, label in runs:
        data = np.load(path)
        arr = data[args.metric].astype(float)          # (seeds, episodes)
        smoothed = moving_average(arr.mean(axis=0), window=args.window)
        # moving_average (mode "valid") ngan hon window-1 diem: dich truc x cho dung so episode
        offset = args.window if len(arr[0]) >= args.window else 1
        x = np.arange(len(smoothed)) + offset
        line, = ax.plot(x, smoothed, label=f"{label} ({arr.shape[0]} seed)")

        if args.band and arr.shape[0] > 1:
            per_seed = np.array([moving_average(a, window=args.window) for a in arr])
            ax.fill_between(x, per_seed.min(axis=0), per_seed.max(axis=0),
                            color=line.get_color(), alpha=0.15, linewidth=0)

    if args.hline is not None:
        label = args.hline_label or f"Optimal ({args.hline:g})"
        ax.axhline(args.hline, color="black", linestyle="--", linewidth=1, label=label)

    ax.set_xlabel("Episode")
    if args.metric == "success":
        ax.set_ylabel(f"Success rate (moving avg, window={args.window})")
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylabel(f"Return (moving avg, window={args.window})")
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.set_title(args.title)
    if args.band:
        ax.text(0.99, 0.01, "shaded: min-max across seeds", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray")
    ax.legend()
    ax.grid(alpha=0.3)
    ensure_dir(args.out)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}")


def plot_policy(args):
    with open(args.qtable, "rb") as f:
        q_dict = pickle.load(f)

    env = make_env(args.setting, invalid_action_mode=args.invalid_mode)
    use_mask = (args.invalid_mode == "mask")

    # Voi setting co deadline/express, mot o (row,col,carrying) duoc tham
    # tai NHIEU gia tri steps_remaining/express khac nhau tuy duong di.
    # Thay vi co dinh steps_remaining=max_steps (hau nhu khong duoc tham
    # o hau het cac o, dan den Q toan so 0 va policy sai), ta gom tat ca
    # cac entry Q ung voi cung (row,col,carrying) lai va lay trung binh -
    # cho ra policy dai dien hop ly hon de truc quan hoa.
    agg = defaultdict(list)
    for state, qvec in q_dict.items():
        key = state[:3]
        agg[key].append(np.asarray(qvec, dtype=np.float64))

    def get_mean_q(row, col, carrying):
        key = (row, col, carrying)
        if key not in agg:
            return None  # o nay chua tung duoc tham trong qua trinh training
        return np.mean(agg[key], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(6 * env.n_cols / 5, 3 * env.n_rows / 5))
    titles = ["carrying = 0 (chua lay hang)", "carrying = 1 (dang mang hang)"]

    for carrying, ax in zip([0, 1], axes):
        ax.set_xlim(-0.5, env.n_cols - 0.5)
        ax.set_ylim(-0.5, env.n_rows - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(env.n_cols))
        ax.set_yticks(range(env.n_rows))
        ax.set_title(titles[carrying])

        for r in range(env.n_rows):
            for c in range(env.n_cols):
                if env.is_wall(r, c):
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
                    continue
                if (r, c) == env.pickup_pos:
                    ax.text(c, r, "P", ha="center", va="center", color="green", fontsize=12, fontweight="bold")
                elif (r, c) == env.dest_pos:
                    ax.text(c, r, "D", ha="center", va="center", color="blue", fontsize=12, fontweight="bold")
                elif env.enable_express and (r, c) == env.express_pos:
                    ax.text(c, r, "E", ha="center", va="center", color="orange", fontsize=12, fontweight="bold")

                q_values = get_mean_q(r, c, carrying)
                if q_values is None:
                    # o nay chua tung duoc tham trong training -> khong ve mui ten,
                    # chi danh dau cham xam de phan biet voi o co du lieu
                    ax.plot(c, r, marker=".", color="lightgray", markersize=4)
                    continue

                mask = env.valid_action_mask((r, c, carrying)) if use_mask else None
                if mask is not None:
                    q_values = np.where(mask, q_values, -np.inf)
                best_action = int(np.argmax(q_values))

                if best_action in ARROW:
                    dx, dy = ARROW[best_action]
                    ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
                else:
                    label = "PICK" if best_action == 4 else "DROP"
                    ax.text(c, r, label, ha="center", va="center", color="purple", fontsize=7)

        ax.set_aspect("equal")

    fig.suptitle(f"Policy cuoi cung - Setting {args.setting}")
    ensure_dir(args.out)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    lc = sub.add_parser("learning-curve")
    lc.add_argument("--npz", type=str, required=True)
    lc.add_argument("--npz2", type=str, default=None)
    lc.add_argument("--npz3", type=str, default=None)
    lc.add_argument("--label1", type=str, default="Run 1")
    lc.add_argument("--label2", type=str, default="Run 2")
    lc.add_argument("--label3", type=str, default="Run 3")
    lc.add_argument("--metric", type=str, default="returns", choices=["returns", "success"],
                    help="returns = return theo episode; success = ti le giao thanh cong theo episode")
    lc.add_argument("--hline", type=float, default=None, help="ve duong ngang (vi du return toi uu)")
    lc.add_argument("--hline-label", type=str, default=None)
    lc.add_argument("--band", action="store_true", help="to bong vung min-max giua cac seed")
    lc.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="gioi han truc y, vi du --ylim 20 40 de phong to vung da hoi tu")
    lc.add_argument("--title", type=str, default="Learning curve")
    lc.add_argument("--window", type=int, default=100)
    lc.add_argument("--out", type=str, required=True)

    pol = sub.add_parser("policy")
    pol.add_argument("--setting", type=int, required=True, choices=[1, 2, 3, 4])
    pol.add_argument("--qtable", type=str, required=True)
    pol.add_argument("--invalid-mode", type=str, default="penalty", choices=["penalty", "mask"])
    pol.add_argument("--out", type=str, required=True)

    args = parser.parse_args()
    if args.command == "learning-curve":
        plot_learning_curve(args)
    elif args.command == "policy":
        plot_policy(args)


if __name__ == "__main__":
    main()
