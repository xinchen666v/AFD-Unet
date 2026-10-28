# -*- coding: utf-8 -*-
"""对比：旧排序法 vs 新通道索引法 生成的 Voronoi LUT（t1 出错样本）"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from scipy.spatial import KDTree
from model import get_model

IMG = r"test/floodmap-npy_1_3_260812/floodmap_28_2.npy"
GT_PATH = r"data/processed/coords/floodmap_28_2_coords.npy"

dev = torch.device("cuda")
model = get_model(dev, load_pretrained=False)
model.load_state_dict(torch.load(r"checkpoints/best_model_finetune.pth", map_location="cpu"))
model.eval()

img = np.load(IMG).astype(np.float32)
img = img / (np.max(img) + 1e-8)
with torch.no_grad():
    hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()
gt = np.load(GT_PATH).astype(np.float32)

MS, R = 512 / 3.0, 85
pts_raw = np.zeros((36, 2)); conf = np.zeros(36)
for ch in range(36):
    m = ch // 4; i, j = m // 3, m % 3
    cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
    x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
    y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
    ry, rx = np.unravel_index(np.argmax(hm[ch, y0:y1, x0:x1]), (y1 - y0, x1 - x0))
    pts_raw[ch] = [x0 + rx, y0 + ry]; conf[ch] = hm[ch, y0:y1, x0:x1].max()

# 旧：按 x 排序分列
idx_x = np.argsort(pts_raw[:, 0])
old_order = []
for c in range(6):
    col = idx_x[c * 6:(c + 1) * 6]
    old_order.extend(col[np.argsort(pts_raw[col, 1])[::-1]])
old_pts = pts_raw[np.array(old_order)]

# 新：通道索引直接映射
new_order = np.zeros(36, dtype=int)
for ch in range(36):
    m = ch // 4; i, j = m // 3, m % 3; k = ch % 4
    row = i * 2 + (0 if k < 2 else 1)
    col = j * 2 + (0 if k % 2 == 0 else 1)
    new_order[col * 6 + (6 - row) - 1] = ch
new_pts = pts_raw[new_order]

print("编号错位（旧 vs 新）:")
bad = [r + 1 for r in range(36) if old_order[r] != new_order[r]]
print("  ", bad, f"共 {len(bad)} 个")

def lut(points):
    gy, gx = np.mgrid[0:512, 0:512]
    tree = KDTree(points)
    _, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    return idx.reshape(512, 512) + 1

cmap = mcolors.ListedColormap(plt.cm.hsv(np.linspace(0, 1, 36)))
vmax = np.percentile(img, 99.5) or 1.0

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

axes[0].imshow(img, cmap="gray", vmin=0, vmax=vmax)
axes[0].scatter(new_pts[:, 0], new_pts[:, 1], c="red", s=45, marker="x", linewidths=1.6, label="pred")
axes[0].scatter(gt[:, 0], gt[:, 1], facecolors="none", edgecolors="lime", s=90, linewidths=1.2, label="GT")
d = np.hypot(*(pts_raw - gt).T)
axes[0].set_title(f"Pred vs GT (channel order)  mean={d.mean():.2f}px  max={d.max():.2f}px", fontsize=13)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].axis("off")

for ax, pts, title in [(axes[1], old_pts, f"OLD: sort by x  ({len(bad)} ids wrong)"),
                       (axes[2], new_pts, "NEW: channel index  (0 wrong)")]:
    ax.imshow(lut(pts), cmap=cmap, interpolation="nearest", vmin=1, vmax=36)
    for k, (x, y) in enumerate(pts):
        ax.text(x, y, str(k + 1), color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    ax.set_title(title, fontsize=13)
    ax.axis("off")

plt.tight_layout()
os.makedirs("res", exist_ok=True)
out = r"res/voronoi_old_vs_new.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
