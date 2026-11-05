
from __future__ import annotations
import argparse
import pickle
from collections import defaultdict
import numpy as np

from envs.delivery_env import make_env, ACTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--qtable", type=str, required=True)
    parser.add_argument("--invalid-mode", type=str, default="penalty", choices=["penalty", "mask"])
    parser.add_argument("--max-render-steps", type=int, default=60)
    args = parser.parse_args()

    with open(args.qtable, "rb") as f:
        q_dict = pickle.load(f)
    Q = defaultdict(lambda: np.zeros(6, dtype=np.float64), q_dict)

    env = make_env(args.setting, invalid_action_mode=args.invalid_mode)
    use_mask = (args.invalid_mode == "mask")

    state, info = env.reset()
    print(env.render())
    print("-" * 40)

    total_reward = 0.0
    for t in range(args.max_render_steps):
        mask = info["action_mask"] if use_mask else None
        q_values = Q[state]
        if mask is not None:
            q_values = np.where(mask, q_values, -np.inf)
        action = int(np.argmax(q_values))

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"t={t:02d} action={ACTIONS[action]:7s} reward={reward:+.1f} "
              f"total_reward={total_reward:+.1f}")
        print(env.render())
        print("-" * 40)

        state = next_state
        if terminated:
            print("KET QUA: Giao hang THANH CONG.")
            break
        if truncated:
            print("KET QUA: THAT BAI (het deadline / timeout).")
            break
    else:
        print("KET QUA: Dat gioi han render steps ma chua terminal.")


if __name__ == "__main__":
    main()
