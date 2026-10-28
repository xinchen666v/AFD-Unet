# -*- coding: utf-8 -*-
"""
用微调后的模型反向筛查训练集中的 bad 标签：
模型定位已很准（正常 <5px），若某张图 GT 与预测偏差很大，多半是 GT 标错了
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
from tqdm import tqdm
from model import get_model

IMG_DIR = r"data/processed/finetune_npy_images"
CRD_DIR = r"data/processed/finetune_coords"
CKPT = r"checkpoints/best_model_finetune.pth"

# 与 train.py 一致的分组划分，用于区分 train / val
import re
GROUP_RE = re.compile(r"^floodmap_(?:(\d{6})_)?(\d+)_(\d+)\.npy$")
def gid(f):
    return int(GROUP_RE.match(f).group(2))

files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".npy")])
groups = [gid(f) for f in files]
uniq = sorted(set(groups))
rng = np.random.default_rng(42)
sh = rng.permutation(len(uniq))
ntr = int(round(0.8 * len(uniq)))
train_g = {uniq[k] for k in sh[:ntr]}

dev = torch.device("cuda")
model = get_model(dev, load_pretrained=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

MS, R = 512 / 3.0, 85
rows = []
for f in tqdm(files, desc="筛查中"):
    img = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)
    img = img / (np.max(img) + 1e-8)
    gt = np.load(os.path.join(CRD_DIR, f.replace(".npy", "_coords.npy"))).astype(np.float32)
    with torch.no_grad():
        hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()

    pts = np.zeros((36, 2)); conf = np.zeros(36)
    for ch in range(36):
        m = ch // 4; i, j = m // 3, m % 3
        cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
        x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
        y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
        ry, rx = np.unravel_index(np.argmax(hm[ch, y0:y1, x0:x1]), (y1 - y0, x1 - x0))
        pts[ch] = [x0 + rx, y0 + ry]; conf[ch] = hm[ch, y0:y1, x0:x1].max()

    d = np.hypot(*(pts - gt).T)
    rows.append(dict(f=f, mean=float(d.mean()), mx=float(d.max()),
                     n10=int((d > 10).sum()), n20=int((d > 20).sum()),
                     conf=float(conf.min()), split="train" if gid(f) in train_g else "val"))

rows.sort(key=lambda r: -r["mx"])
print(f"\n{'排名':<4}{'文件':<34}{'mean':>7}{'max':>7}{'>10px':>7}{'>20px':>7}{'minConf':>9}  split")
for k, r in enumerate(rows[:20], 1):
    print(f"{k:<4}{r['f']:<34}{r['mean']:7.2f}{r['mx']:7.2f}{r['n10']:7d}{r['n20']:7d}{r['conf']:9.2f}  {r['split']}")

mx = np.array([r["mx"] for r in rows])
mean = np.array([r["mean"] for r in rows])
print(f"\n=== 全量 384 张统计 ===")
print(f"  mean 误差: 中位 {np.median(mean):.2f}px, p95 {np.percentile(mean,95):.2f}px")
print(f"  max  误差: 中位 {np.median(mx):.2f}px, p95 {np.percentile(mx,95):.2f}px")
for th in [10, 15, 20, 30]:
    n = sum(1 for r in rows if r["mx"] > th)
    print(f"  存在 >{th}px 点的图: {n} 张 ({n/len(rows)*100:.1f}%)")
