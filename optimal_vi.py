
import argparse
from envs.delivery_env import make_env


def set_state(env, s):
    env._row, env._col, env._carrying = s[0], s[1], s[2]
    i = 3
    if env.enable_deadline:
        env._steps_remaining = s[i]
        i += 1
    if env.enable_express:
        env._express_timer, env._express_used = s[i], s[i + 1]


def solve(setting, gamma):
    env = make_env(setting)
    s0, _ = env.reset()

    # 1) Duyet cac state co the den duoc (reachable) va ghi lai transition tat dinh
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

    # 2) Value Iteration: v(s) <- max_a [ r + gamma * v(s') ]
    V = {s: 0.0 for s in trans}
    n_iter = 0
    for n_iter in range(1, 5001):
        delta = 0.0
        for s in trans:
            best = max(r + (0.0 if done else gamma * V[ns]) for ns, r, done in trans[s])
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < 1e-10:
            break

    # 3) Chay thu policy tham lam theo v*, cong reward (khong chiet khau)
    s, ret, steps, used_express = s0, 0.0, 0, False
    while steps < 200:
        a = max(range(env.n_actions),
                key=lambda a: trans[s][a][1] + (0.0 if trans[s][a][2] else gamma * V[trans[s][a][0]]))
        ns, r, done = trans[s][a]
        ret += r
        steps += 1
        if env.enable_express and ns[-1] == 1:
            used_express = True
        s = ns
        if done:
            break
    return env.n_states, len(trans), n_iter, ret, steps, used_express


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    print("| Setting | |S| ly thuyet | |S| reachable | So vong VI | Return toi uu | So buoc | Dung express |")
    print("|---|---|---|---|---|---|---|")
    for st in [1, 2, 3, 4]:
        upper, reach, n_iter, ret, steps, used = solve(st, args.gamma)
        print(f"| {st} | {upper} | {reach} | {n_iter} | {ret:.1f} | {steps} | {'co' if used else 'khong'} |")


if __name__ == "__main__":
    main()
