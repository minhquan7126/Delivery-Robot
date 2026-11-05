"""
Tong hop ket qua training thanh bang markdown de dua vao bao cao.

Chay (dung o thu muc goc delivery_robot):
    python summarize.py
    python summarize.py --dir results --last 200

Voi moi file results/*.npz, in ra:
  - return trung binh cua --last episode cuoi (tung seed)
  - success rate cua --last episode cuoi (tung seed)
  - return trung binh 100 episode DAU (the hien muc phat luc chua hoc)
  - so episode can de return trung binh truot (cua so 100) dat >= 0  (toc do hoi tu)
  - so episode can de success rate trung binh truot (cua so --last) dat >= 90%
LUU Y: o setting 1, 2 (khong co deadline) episode chi ket thuc khi giao thanh cong
nen success rate luon = 1.00; hay dung return de danh gia o hai setting nay.
"""
import argparse
import glob
import os
import numpy as np


def first_reach(values, window, thr):
    """Episode dau tien ma trung binh truot (cua so window) cua values >= thr."""
    if len(values) < window:
        return None
    ma = np.convolve(values, np.ones(window) / window, mode="valid")
    idx = np.where(ma >= thr)[0]
    return int(idx[0] + window) if len(idx) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    parser.add_argument("--last", type=int, default=200)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not files:
        print("Khong tim thay file .npz trong", args.dir)
        return

    print(f"| Run | Return {args.last} ep cuoi (moi seed) | Success {args.last} ep cuoi (moi seed) "
          f"| Return 100 ep dau | Episode return>=0 | Episode success>=90% |")
    print("|---|---|---|---|---|---|")
    for path in files:
        name = os.path.basename(path)[:-4]
        data = np.load(path)
        returns = data["returns"]            # (seeds, episodes)
        success = data["success"].astype(float)
        ret_last = returns[:, -args.last:].mean(axis=1)
        suc_last = success[:, -args.last:].mean(axis=1)
        ret_first = returns[:, :100].mean()
        reach = first_reach(success.mean(axis=0), args.last, 0.9)
        reach_ret = first_reach(returns.mean(axis=0), 100, 0.0)
        print(f"| {name} | {' / '.join(f'{x:.2f}' for x in ret_last)} "
              f"| {' / '.join(f'{x:.2f}' for x in suc_last)} "
              f"| {ret_first:.1f} | {reach_ret if reach_ret is not None else 'chua dat'} "
              f"| {reach if reach is not None else 'chua dat'} |")


if __name__ == "__main__":
    main()
