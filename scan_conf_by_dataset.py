# -*- coding: utf-8 -*-
"""对比三个数据集上模型的最低通道置信度，量化"模糊图"到底差多少"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
from tqdm import tqdm
from model import get_model

DIRS = {
    "1-7    (模糊/噪点多)": r"test/floodmap-npy1-7",
    "1-18   (已用于微调)": r"test/floodmap-npy1-18",
    "261020 (t2, 全对)":  r"test/floodmap-npy_1_1_261020",
}

dev = torch.device("cuda")
model = get_model(dev, load_pretrained=False)
model.load_state_dict(torch.load(r"checkpoints/best_model_finetune.pth", map_location="cpu"))
model.eval()

MS, R = 512 / 3.0, 85

def min_conf_of(path):
    img = np.load(path).astype(np.float32)
    img = img / (np.max(img) + 1e-8)
    with torch.no_grad():
        hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()
    confs = []
    for ch in range(36):
        m = ch // 4; i, j = m // 3, m % 3
        cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
        x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
        y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
        confs.append(hm[ch, y0:y1, x0:x1].max())
    return float(np.min(confs)), float(np.mean(confs))

print(f"{'数据集':<22}{'张数':>5}{'minConf中位':>12}{'meanConf中位':>13}{'<0.5':>7}{'<0.6':>7}{'<0.7':>7}{'<0.8':>7}")
for name, d in DIRS.items():
    files = sorted([f for f in os.listdir(d) if f.endswith(".npy")])
    res = [min_conf_of(os.path.join(d, f)) for f in tqdm(files, desc=name, leave=False)]
    mins = np.array([r[0] for r in res]); means = np.array([r[1] for r in res])
    print(f"{name:<22}{len(files):>5}{np.median(mins):>12.3f}{np.median(means):>13.3f}"
          f"{(mins<0.5).sum():>7}{(mins<0.6).sum():>7}{(mins<0.7).sum():>7}{(mins<0.8).sum():>7}")
