# -*- coding: utf-8 -*-
"""
C++ 输出 vs Python 参考基准 的逐项对比 + 可视化（参数化）

用法:
  python verify_cpp_output.py --prefix floodmap_1_5
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_csv(path):
    out = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a = line.strip().split(",")
            if len(a) == 4:
                out.append((int(a[0]), float(a[1]), float(a[2]), float(a[3])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"cpp_demo/test_data")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tag", default="", help="对比图标题附加信息")
    args = ap.parse_args()

    D, P = args.dir, args.prefix
    ref = load_csv(os.path.join(D, f"ref_{P}_points.csv"))
    cpp = load_csv(os.path.join(D, f"cpp_{P}_points.csv"))
    print(f"点数: Python={len(ref)}  C++={len(cpp)}")

    diffs = []
    for a, b in zip(ref, cpp):
        d = float(np.hypot(a[1] - b[1], a[2] - b[2]))
        diffs.append((a[0], a[1:3], b[1:3], d, abs(a[3] - b[3])))

    bad = [t for t in diffs if t[3] > 0]
    max_d = max(t[3] for t in diffs)
    max_dc = max(t[4] for t in diffs)

    print(f"坐标不一致: {len(bad)}/36   max|Δxy| = {max_d:.2f} px")
    print(f"置信度 max|Δconf| = {max_dc:.3e}")
    for t in bad[:8]:
        print(f"  #{t[0]}: py={t[1]} cpp={t[2]}  dist={t[3]:.1f}")

    lut_ref = np.fromfile(os.path.join(D, f"ref_{P}_lut.raw"), dtype=np.uint8).reshape(512, 512)
    lut_cpp = np.fromfile(os.path.join(D, f"cpp_{P}_lut.raw"), dtype=np.uint8).reshape(512, 512)
    diff = lut_ref != lut_cpp
    n = int(diff.sum())
    print(f"LUT 不一致: {n} / {diff.size} ({n / diff.size * 100:.4f}%)")

    # ---- 画图 ----
    img = np.fromfile(os.path.join(D, f"{P}.bin"), dtype=np.float32).reshape(512, 512)
    img = img / (img.max() + 1e-8)
    cmap = mcolors.ListedColormap(plt.cm.hsv(np.linspace(0, 1, 36)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
    axes[0].scatter([p[1] for p in ref], [p[2] for p in ref],
                    s=90, facecolors="none", edgecolors="lime", linewidths=1.5, label="Python")
    axes[0].scatter([p[1] for p in cpp], [p[2] for p in cpp],
                    c="red", s=22, marker="x", linewidths=1.2, label="C++")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].set_title(f"{P}   max|Δ|={max_d:.2f} px", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(lut_ref, cmap=cmap, vmin=1, vmax=36, interpolation="nearest")
    axes[1].set_title("Python LUT (KDTree)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(lut_cpp, cmap=cmap, vmin=1, vmax=36, interpolation="nearest")
    axes[2].set_title(f"C++ LUT (brute force)   {n} px diff", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    out = os.path.join(D, f"compare_{P}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n对比图: {out}")

    ok = (len(bad) == 0) and (n / diff.size < 0.005)
    print(f"结论: {'通过' if ok else '需检查'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
