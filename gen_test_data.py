# -*- coding: utf-8 -*-
"""
生成 C++ 测试数据与 Python 参考基准（参数化）

用法:
  python gen_test_data.py --img test/floodmap-npy_1_1_261020/floodmap_1_5.npy
  python gen_test_data.py --img xxx.npy --out cpp_demo/test_data --prefix mycase
"""
import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from scipy.spatial import KDTree
from model import get_model

CKPT = r"checkpoints/best_model_finetune_blur.pth"
MS, R = 512 / 3.0, 85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="输入 .npy 路径")
    ap.add_argument("--out", default=r"cpp_demo/test_data", help="输出目录")
    ap.add_argument("--prefix", default=None, help="输出前缀，默认取输入文件名")
    args = ap.parse_args()

    prefix = args.prefix or os.path.splitext(os.path.basename(args.img))[0]
    os.makedirs(args.out, exist_ok=True)

    # ---- 1. 原始数据 -> .bin ----
    img = np.load(args.img).astype(np.float32)
    assert img.shape == (512, 512), f"尺寸不符: {img.shape}"
    img.tofile(os.path.join(args.out, f"{prefix}.bin"))
    print(f"[bin] {prefix}.bin  max={img.max():.2f}  min={img.min():.2f}")

    # ---- 2. PyTorch 参考推理 ----
    model = get_model(torch.device("cpu"), load_pretrained=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()

    x = img / (np.max(img) + 1e-8)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None, None]).numpy()[0]
    hm = 1.0 / (1.0 + np.exp(-logits))

    # ---- 3. 36 通道 ROI 取峰 ----
    pts_ch = np.zeros((36, 2))
    conf_ch = np.zeros(36)
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
        x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
        y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
        roi = hm[ch, y0:y1, x0:x1]
        ry, rx = np.unravel_index(np.argmax(roi), roi.shape)
        pts_ch[ch] = [x0 + rx, y0 + ry]
        conf_ch[ch] = roi.max()

    # ---- 4. 通道重排 -> LUT 编号 ----
    order = np.zeros(36, dtype=int)
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        k = ch % 4
        row = i * 2 + (0 if k < 2 else 1)
        col = j * 2 + (0 if k % 2 == 0 else 1)
        order[col * 6 + (6 - row) - 1] = ch
    pts = pts_ch[order]
    conf = conf_ch[order]

    csv_path = os.path.join(args.out, f"ref_{prefix}_points.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,x,y,conf\n")
        for k in range(36):
            f.write(f"{k + 1},{pts[k, 0]:.1f},{pts[k, 1]:.1f},{conf[k]:.6f}\n")
    print(f"[csv] ref_{prefix}_points.csv  min_conf={conf.min():.4f}")

    # ---- 5. Voronoi LUT ----
    gy, gx = np.mgrid[0:512, 0:512]
    tree = KDTree(pts)
    _, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    lut = (idx.reshape(512, 512) + 1).astype(np.uint8)
    lut.tofile(os.path.join(args.out, f"ref_{prefix}_lut.raw"))
    print(f"[lut] ref_{prefix}_lut.raw  unique={np.unique(lut).size}")

    print("\n前 5 个点:")
    for k in range(5):
        print(f"  #{k + 1}: ({pts[k, 0]:.0f}, {pts[k, 1]:.0f})  conf={conf[k]:.4f}")


if __name__ == "__main__":
    main()
