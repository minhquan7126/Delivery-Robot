
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from envs.delivery_env import make_env
from dp_solvers import build_model, make_reward_config

ARROW = {0: (0, -0.25), 1: (0, 0.25), 2: (-0.25, 0), 3: (0.25, 0)}  # Up, Down, Left, Right
LABEL = {4: "PICK", 5: "DROP"}


# --------------------------------------------------------------------------- mo hinh dang ma tran
class Model:
    def __init__(self, trans, gamma):
        self.states = list(trans.keys())
        self.idx = {s: i for i, s in enumerate(self.states)}
        self.n = len(self.states)
        A = len(trans[self.states[0]])
        self.A = A
        self.gamma = gamma
        self.NS = np.zeros((self.n, A), dtype=np.int64)
        self.R = np.zeros((self.n, A))
        self.D = np.zeros((self.n, A), dtype=bool)
        for s, i in self.idx.items():
            for a, (ns, r, done) in enumerate(trans[s]):
                self.R[i, a] = r
                self.D[i, a] = done
                self.NS[i, a] = i if done else self.idx[ns]

    def q(self, V):
        return self.R + self.gamma * np.where(self.D, 0.0, V[self.NS])


def solve_vstar(m, tol=1e-12, max_iter=100000):
    V = np.zeros(m.n)
    for _ in range(max_iter):
        new = m.q(V).max(axis=1)
        done = np.max(np.abs(new - V)) < tol
        V = new
        if done:
            break
    return V


def evaluate(m, policy, V, n_eval=None, tol=1e-12, cap=50000):
    """Policy evaluation: n_eval sweep (hoac den hoi tu neu n_eval=None). Tra ve (V, so sweep)."""
    rows = np.arange(m.n)
    Rp, NSp, Dp = m.R[rows, policy], m.NS[rows, policy], m.D[rows, policy]
    sweeps = 0
    for _ in range(cap if n_eval is None else n_eval):
        new = Rp + m.gamma * np.where(Dp, 0.0, V[NSp])
        delta = np.max(np.abs(new - V))
        V = new
        sweeps += 1
        if n_eval is None and delta < tol:
            break
    return V, sweeps


def truncated_pi(m, Vstar, n_eval, max_iter=60, stop=1e-6):
    """n_eval = so sweep evaluation moi iteration (None = den hoi tu = Policy Iteration)."""
    V = np.zeros(m.n)
    errs = [np.max(np.abs(V - Vstar))]            # dung de xac dinh k hoi tu (sai so lon nhat)
    merrs = [np.mean(np.abs(V - Vstar))]          # dung de ve (sai so trung binh, de doc hon)
    means = [V.mean()]
    sweeps = 0
    for _ in range(max_iter):
        policy = m.q(V).argmax(axis=1)               # policy improvement
        V, sw = evaluate(m, policy, V, n_eval)       # (truncated) policy evaluation
        sweeps += sw
        errs.append(np.max(np.abs(V - Vstar)))
        merrs.append(np.mean(np.abs(V - Vstar)))
        means.append(V.mean())
        if errs[-1] < stop:
            break
    return dict(errs=np.array(errs), merrs=np.array(merrs), means=np.array(means), sweeps=sweeps,
                converged=errs[-1] < stop)


def policy_iteration_history(m, init_action=0, max_rounds=200):
    """Policy Iteration chuan: tra ve list [(pi_k, v_pi_k)], pi_0 = luon chon init_action."""
    rows = np.arange(m.n)
    policy = np.full(m.n, init_action, dtype=np.int64)
    V = np.zeros(m.n)
    hist = []
    for _ in range(max_rounds):
        V, _ = evaluate(m, policy, V, None)
        hist.append((policy.copy(), V.copy()))
        q = m.q(V)
        new = np.where(q.max(axis=1) > q[rows, policy] + 1e-9, q.argmax(axis=1), policy)
        if np.array_equal(new, policy):
            break
        policy = new
    return hist


# --------------------------------------------------------------------------- Hinh 1
def plot_policy_rounds(env, m, hist, n_show, out, title, rows_per_fig=2):
    """Ve cac round PI. Moi hinh chua toi da rows_per_fig round (de vua trang Word, chu doc duoc)."""
    picks = sorted(set(int(x) for x in np.linspace(0, len(hist) - 1, min(n_show, len(hist))).round()))
    cell = {}
    for i, s in enumerate(m.states):
        key = s[:3]
        score = s[3] if len(s) > 3 else 0
        if key not in cell or score > cell[key][0]:
            cell[key] = (score, i)

    nr, nc = env.n_rows, env.n_cols
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgray")
    chunks = [picks[i:i + rows_per_fig] for i in range(0, len(picks), rows_per_fig)]

    for ci, chunk in enumerate(chunks):
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6.6, 3.1 * len(chunk) + 0.4), squeeze=False)
        for ri, k in enumerate(chunk):
            pol, V = hist[k]
            for carrying in (0, 1):
                ax = axes[ri][carrying]
                grid = np.full((nr, nc), np.nan)
                for r in range(nr):
                    for c in range(nc):
                        if not env.is_wall(r, c) and (r, c, carrying) in cell:
                            grid[r, c] = V[cell[(r, c, carrying)][1]]
                ax.imshow(np.ma.masked_invalid(grid), cmap=cmap)
                for r in range(nr):
                    for c in range(nc):
                        if env.is_wall(r, c):
                            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
                            continue
                        if (r, c) == env.pickup_pos:
                            ax.text(c - 0.45, r - 0.28, "P", fontsize=8, color="green", fontweight="bold")
                        elif (r, c) == env.dest_pos:
                            ax.text(c - 0.45, r - 0.28, "D", fontsize=8, color="blue", fontweight="bold")
                        elif env.enable_express and (r, c) == env.express_pos:
                            ax.text(c - 0.45, r - 0.28, "E", fontsize=8, color="darkorange", fontweight="bold")
                        if (r, c, carrying) not in cell:
                            continue
                        i = cell[(r, c, carrying)][1]
                        ax.text(c, r + 0.38, f"{V[i]:.1f}", ha="center", va="center", fontsize=5.5)
                        a = int(pol[i])
                        if a in ARROW:
                            dx, dy = ARROW[a]
                            ax.annotate("", xy=(c + dx * 1.6, r - 0.05 + dy * 1.6), xytext=(c, r - 0.05),
                                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
                        else:
                            ax.text(c, r - 0.05, LABEL[a], ha="center", va="center",
                                    fontsize=6, color="purple", fontweight="bold")
                ax.set_xticks(range(nc))
                ax.set_yticks(range(nr))
                ax.tick_params(labelsize=7)
                if ri == 0:
                    ax.set_title("carrying = 0" if carrying == 0 else "carrying = 1", fontsize=9)
                if carrying == 0:
                    ax.set_ylabel(f"$\\pi_{{{k}}}$ and $v_{{\\pi_{{{k}}}}}$", fontsize=10)
        fig.suptitle(title, fontsize=9, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        path = out if len(chunks) == 1 else out.replace(".png", f"_part{ci + 1}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {path}  (cac round: {chunk}; tong {len(hist)} round)")


# --------------------------------------------------------------------------- Hinh 2
def plot_convergence(res, Vstar, out, log, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    styles = [("Policy iteration", "tab:blue", "o"),
              ("Value iteration", "tab:purple", "D"),
              ("Truncated policy iteration", "black", "s")]
    for i, ((name, color, mk), r) in enumerate(zip(styles, res)):
        ks = np.arange(len(r["means"]))
        k_conv = len(ks) - 1 if r["converged"] else None   # k dau tien co sai so < 1e-6
        lab = (f"{name} (k = {k_conv})" if k_conv is not None
               else f"{name} (chua hoi tu sau k = {len(ks) - 1})")
        a1.plot(ks, r["means"], color=color, marker=mk, ms=4, label=lab)
        a2.plot(ks, np.maximum(r["merrs"], 1e-12), color=color, marker=mk, ms=4, label=lab)
        if k_conv is not None:
            for ax in (a1, a2):
                ax.axvline(k_conv, color=color, ls="--", lw=1, alpha=0.7)
                ax.text(k_conv, 0.92 - 0.07 * i, f" k={k_conv}", color=color, fontsize=9,
                        fontweight="bold", transform=ax.get_xaxis_transform(), va="bottom")
    xmax = max(len(r["means"]) - 1 for r in res)
    for ax in (a1, a2):
        ax.set_xlim(0, xmax * 1.15 + 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    a1.axhline(Vstar.mean(), color="brown", label="Optimal state value $v^*$")
    a1.set_xlabel("k")
    a1.set_ylabel("trung binh state value $V_k$")
    a1.set_title("Gia tri $V_k$ tien den $v^*$")
    a2.set_xlabel("k")
    a2.set_ylabel("sai so TB $\\frac{1}{|S|}\\sum_s |V_k(s) - v^*(s)|$")
    a2.set_title("Sai so trung binh so voi $v^*$")
    if log:
        a2.set_yscale("log")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------- Hinh 3
def plot_truncated(xs, runs, out, log, title):
    fig, axes = plt.subplots(len(xs), 1, figsize=(7, 1.9 * len(xs) + 0.8), sharex=True, squeeze=False)
    xmax = max(len(r["errs"]) for r in runs)
    for ax, x, r in zip(axes[:, 0], xs, runs):
        ax.plot(np.maximum(r["merrs"], 1e-12) if log else r["merrs"], marker="o", ms=3,
                color="tab:blue", label=f"Truncated policy iteration-{x}")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(0, xmax)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("iteration num")
    fig.supylabel("state value error (TB |V_k - v*|)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-config", type=str, default="default",
                        choices=["default", "low_invalid_penalty", "high_step_penalty"])
    parser.add_argument("--outdir", type=str, default="figs")
    parser.add_argument("--max-iter", type=int, default=60, help="so iteration toi da cho hinh 3")
    parser.add_argument("--conv-iter", type=int, default=1000,
                        help="so iteration toi da cho hinh 2 (de VI chay het den luc hoi tu, lay k chinh xac)")
    parser.add_argument("--xs", type=int, nargs="+", default=[1, 3, 6, 100],
                        help="so sweep evaluation cua truncated PI (hinh 3)")
    parser.add_argument("--trunc", type=int, default=5, help="x cua truncated PI o hinh 2")
    parser.add_argument("--rounds", type=int, default=4, help="so round PI hien thi o hinh 1")
    parser.add_argument("--log", action="store_true", help="truc y log cho hinh 2 va 3")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    env = make_env(args.setting, reward_config=make_reward_config(args.reward_config))
    s0, trans = build_model(env)
    m = Model(trans, args.gamma)
    Vstar = solve_vstar(m)
    tag = f"s{args.setting}_{args.reward_config}"
    head = f"Setting {args.setting}, gamma={args.gamma}, |S| reachable={m.n}"
    print(head)

    # Hinh 1
    hist = policy_iteration_history(m)
    plot_policy_rounds(env, m, hist, args.rounds, os.path.join(args.outdir, f"{tag}_fig1_pi_rounds.png"),
                       f"Policy iteration - cac policy/value trung gian ({head})")

    # Hinh 2
    res = [truncated_pi(m, Vstar, None, args.conv_iter),
           truncated_pi(m, Vstar, 1, args.conv_iter),
           truncated_pi(m, Vstar, args.trunc, args.conv_iter)]
    plot_convergence(res, Vstar, os.path.join(args.outdir, f"{tag}_fig2_convergence.png"), args.log,
                     f"PI vs VI vs Truncated PI-{args.trunc} ({head})")
    print("\nHinh 2 - k chinh xac (k dau tien co max|V_k - v*| < 1e-6):")
    print("| Thuat toan | k hoi tu | Tong so sweep evaluation |")
    print("|---|---|---|")
    for name, r in zip(["Policy iteration", "Value iteration", f"Truncated PI-{args.trunc}"], res):
        k = len(r["errs"]) - 1
        print(f"| {name} | {k if r['converged'] else f'chua hoi tu sau {k}'} | {r['sweeps']} |")

    # Hinh 3
    runs, rows = [], []
    for x in args.xs:
        t0 = time.time()
        r = truncated_pi(m, Vstar, x, args.max_iter)
        r["time"] = time.time() - t0
        runs.append(r)
        rows.append((str(x), r))
    plot_truncated(args.xs, runs, os.path.join(args.outdir, f"{tag}_fig3_truncated.png"), args.log,
                   f"Truncated policy iteration-x ({head})")

    # Bang so lieu cho bao cao
    t0 = time.time()
    pi_run = truncated_pi(m, Vstar, None, args.max_iter)
    pi_run["time"] = time.time() - t0
    rows.append(("inf (PI)", pi_run))
    print("\n| x (sweep/iteration) | So iteration den sai so < 1e-6 | Tong so sweep | Thoi gian (s) |")
    print("|---|---|---|---|")
    for name, r in rows:
        n_it = len(r["errs"]) - 1
        print(f"| {name} | {n_it if r['converged'] else f'chua hoi tu sau {n_it}'} "
              f"| {r['sweeps']} | {r['time']:.2f} |")


if __name__ == "__main__":
    main()
