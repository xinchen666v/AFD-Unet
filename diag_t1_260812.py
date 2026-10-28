# -*- coding: utf-8 -*-
"""
t1 出错样本诊断：test/floodmap-npy_1_3_260812/floodmap_28_2.npy
1) 生成 36 通道热图
2) 与 GT 逐点对比误差
3) 检查排序法编号是否合理
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import get_model

IMG_PATH = r"test/floodmap-npy_1_3_260812/floodmap_28_2.npy"
GT_PATH = r"data/processed/coords/floodmap_28_2_coords.npy"
CKPT = r"checkpoints/best_model_finetune.pth"

dev = torch.device("cuda")
model = get_model(dev, load_pretrained=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

img = np.load(IMG_PATH).astype(np.float32)
img = img / (np.max(img) + 1e-8)
with torch.no_grad():
    hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()

gt = np.load(GT_PATH).astype(np.float32)
print(f"GT shape: {gt.shape}")

MS = 512 / 3.0
R = 85
pts = np.zeros((36, 2))
conf = np.zeros(36)
for ch in range(36):
    m = ch // 4; i, j = m // 3, m % 3
    cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
    x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
    y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
    ry, rx = np.unravel_index(np.argmax(hm[ch, y0:y1, x0:x1]), (y1 - y0, x1 - x0))
    pts[ch] = [x0 + rx, y0 + ry]
    conf[ch] = hm[ch, y0:y1, x0:x1].max()

err = np.hypot(pts[:, 0] - gt[:, 0], pts[:, 1] - gt[:, 1])
order = np.argsort(err)[::-1]

print("\n=== 误差最大的 12 个通道 ===")
print("通道  预测(x,y)      GT(x,y)      误差(px)   conf")
for ch in order[:12]:
    flag = "  <<<" if err[ch] > 10 else ""
    print(f"Ch{ch:2d}  ({pts[ch,0]:5.1f},{pts[ch,1]:5.1f})  ({gt[ch,0]:5.1f},{gt[ch,1]:5.1f})   {err[ch]:6.1f}    {conf[ch]:.3f}{flag}")

print(f"\n平均误差: {err.mean():.2f} px | 中位数: {np.median(err):.2f} px | 最大: {err.max():.2f} px")
print(f"误差 >10px 的通道数: {(err > 10).sum()}")
print(f"误差 >5px  的通道数: {(err > 5).sum()}")

# --- 排序法编号 vs 通道索引编号 ---
idx_x = np.argsort(pts[:, 0])
rank_sort = np.zeros(36, dtype=int)
for c in range(6):
    col = idx_x[c * 6:(c + 1) * 6]
    y_order = col[np.argsort(pts[col, 1])[::-1]]
    for r, ch in enumerate(y_order):
        rank_sort[ch] = c * 6 + (r + 1)

rank_ch = np.zeros(36, dtype=int)
for ch in range(36):
    m = ch // 4; i, j = m // 3, m % 3; k = ch % 4
    row = i * 2 + (0 if k < 2 else 1)
    col = j * 2 + (0 if k % 2 == 0 else 1)
    rank_ch[ch] = col * 6 + (6 - row)

diff = rank_sort != rank_ch
print(f"\n=== 排序法 vs 通道索引法 编号不一致: {diff.sum()}/36 ===")
for ch in range(36):
    if diff[ch]:
        print(f"  Ch{ch:2d} 排序={rank_sort[ch]:2d} 通道={rank_ch[ch]:2d}  误差={err[ch]:.1f}px")

# --- 生成 36 通道热图 ---
fig, axes = plt.subplots(6, 6, figsize=(15, 15))
axes = axes.flatten()
for ch in range(36):
    ax = axes[ch]
    ax.imshow(hm[ch], cmap="hot", vmin=0, vmax=1)
    mx = float(hm[ch].max())
    color = "#c0392b" if mx < 0.90 else "#2c3e50"
    ax.set_title(f"Ch{ch} (Max: {mx:.2f})", fontsize=11, color=color)
    ax.axis("off")
plt.suptitle("t1 error sample: floodmap-npy_1_3_260812/floodmap_28_2.npy", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.94)
os.makedirs("res", exist_ok=True)
out = r"res/t1_260812_floodmap_28_2_channels.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\n热图已保存: {out}")
