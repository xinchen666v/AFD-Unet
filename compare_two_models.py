# -*- coding: utf-8 -*-
"""对比两代模型在三个数据集上的置信度分布"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
from tqdm import tqdm
from model import get_model

MODELS = {
    "v1 (无模糊图)": r"checkpoints/best_model_finetune.pth",
    "v2 (+192模糊)": r"checkpoints/best_model_finetune_blur.pth",
}
DATASETS = {
    "1-2 独立模糊": r"test/floodmap-npy1-2",
    "261020 清晰":  r"test/floodmap-npy_1_1_261020",
    "1-7 已入训练": r"test/floodmap-npy1-7",
}

dev = torch.device("cuda")
MS, R = 512 / 3.0, 85

def min_conf_all(model, d):
    files = sorted([f for f in os.listdir(d) if f.endswith(".npy")])
    out = []
    for f in tqdm(files, desc=os.path.basename(d), leave=False):
        img = np.load(os.path.join(d, f)).astype(np.float32)
        img = img / (np.max(img) + 1e-8)
        with torch.no_grad():
            hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()
        cs = []
        for ch in range(36):
            m = ch // 4; i, j = m // 3, m % 3
            cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
            x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
            y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
            cs.append(hm[ch, y0:y1, x0:x1].max())
        out.append(float(np.min(cs)))
    return np.array(out)

for tag, ckpt in MODELS.items():
    model = get_model(dev, load_pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    print(f"\n{'='*70}\n模型: {tag}\n{'='*70}")
    print(f"{'数据集':<16}{'minConf中位':>12}{'<0.5':>7}{'<0.6':>7}{'<0.7':>7}{'<0.8':>7}{'<0.9':>7}")
    for dname, d in DATASETS.items():
        v = min_conf_all(model, d)
        print(f"{dname:<16}{np.median(v):>12.3f}{(v<0.5).sum():>7}{(v<0.6).sum():>7}"
              f"{(v<0.7).sum():>7}{(v<0.8).sum():>7}{(v<0.9).sum():>7}")
