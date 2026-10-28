# -*- coding: utf-8 -*-
"""真正的 t1 图 (floodmap-npy1-7) 的 36 通道输出，低置信度标红"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import get_model

dev = torch.device("cuda")
model = get_model(dev, load_pretrained=False)
model.load_state_dict(torch.load(r"checkpoints/best_model_finetune.pth", map_location="cpu"))
model.eval()

PATH = r"test/floodmap-npy1-7/floodmap_28_2.npy"
img = np.load(PATH).astype(np.float32)
img = img / (np.max(img) + 1e-8)
with torch.no_grad():
    hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()

fig, axes = plt.subplots(6, 6, figsize=(15, 15))
axes = axes.flatten()
low = []
for ch in range(36):
    ax = axes[ch]
    ax.imshow(hm[ch], cmap="hot", vmin=0, vmax=1)
    mx = float(hm[ch].max())
    if mx < 0.80:
        low.append(ch)
    ax.set_title(f"Ch{ch} (Max: {mx:.2f})", fontsize=11,
                 color="#c0392b" if mx < 0.80 else "#2c3e50")
    ax.axis("off")

plt.suptitle(f"TRUE t1 file: floodmap-npy1-7/floodmap_28_2.npy   |   red = Max < 0.80 ({len(low)} channels)",
             fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.94)
os.makedirs("res", exist_ok=True)
out = r"res/t1_floodmap_28_2_channels.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
print("低置信度通道(<0.80):", low)
