# -*- coding: utf-8 -*-
"""
评估模型在指定数据集上的定位精度（需要 GT）

提点逻辑与 batch_test_v1.py 完全一致：
  - 3x3 模块 ROI，radius = 85
  - 每通道 ROI 内 argmax
唯一区别：本脚本不做 Voronoi 编号（只比坐标，不需要编号）

用法:
  python eval_position_error.py
  python eval_position_error.py --models a.pth b.pth --data 1-2
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
import re
import torch
import numpy as np
from model import get_model

# ================= 数据集注册表 =================
DATASETS = {
    "1-2":  (r"test/floodmap-npy1-2",
             r"test/2-7-18结果/260305floodmap/floodmap/floodmap_1_2_output/data"),
    "1-7":  (r"test/floodmap-npy1-7",
             r"test/2-7-18结果/260305floodmap/floodmap/floodmap_1_7_output/data"),
    "1-18": (r"test/floodmap-npy1-18",
             r"test/2-7-18结果/260305floodmap/floodmap/floodmap_1_18_output/data"),
}

DEFAULT_MODELS = [
    ("v1 (无模糊图)", "checkpoints/best_model_finetune.pth"),
    ("v2 (+192模糊)", "checkpoints/best_model_finetune_blur.pth"),
]

RE_IMG = re.compile(r"^floodmap_(\d+)_(\d+)\.npy$")
MS, RADIUS = 512 / 3.0, 85


def predict_points(model, img, dev):
    """返回 (36,2) 坐标，顺序 = 通道顺序"""
    img = img / (np.max(img) + 1e-8)
    with torch.no_grad():
        hm = torch.sigmoid(model(torch.from_numpy(img)[None, None].to(dev))).cpu().squeeze(0).numpy()

    pts = np.zeros((36, 2))
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
        x0, x1 = max(0, int(cx - RADIUS)), min(512, int(cx + RADIUS))
        y0, y1 = max(0, int(cy - RADIUS)), min(512, int(cy + RADIUS))
        ry, rx = np.unravel_index(np.argmax(hm[ch, y0:y1, x0:x1]), (y1 - y0, x1 - x0))
        pts[ch] = [x0 + rx, y0 + ry]
    return pts


def evaluate(model, img_dir, lab_dir, dev):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    rows = []
    for f in files:
        m = RE_IMG.match(f)
        if not m:
            continue
        i, j = int(m.group(1)), int(m.group(2))
        gt_path = os.path.join(lab_dir, f"floodmap_a{i:02d}_b{j:02d}_pts.npy")
        if not os.path.isfile(gt_path):
            continue
        gt = np.load(gt_path).astype(np.float32)
        pts = predict_points(model, np.load(os.path.join(img_dir, f)).astype(np.float32), dev)
        d = np.hypot(*(pts - gt).T)
        rows.append((d.mean(), d.max(), int((d > 10).sum()), int((d > 20).sum())))
    return np.array(rows), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="权重路径列表，默认对比 v1/v2")
    ap.add_argument("--data", default="1-2", choices=list(DATASETS.keys()))
    args = ap.parse_args()

    models = ([(os.path.basename(p), p) for p in args.models]
              if args.models else DEFAULT_MODELS)
    img_dir, lab_dir = DATASETS[args.data]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"数据集: {args.data}  ({img_dir})")
    print(f"{'模型':<20}{'mean中位':>10}{'mean p95':>10}{'mean最大':>10}"
          f"{'max中位':>10}{'max p95':>10}{'>10px/图':>10}{'>20px/图':>10}")

    for tag, ckpt in models:
        model = get_model(dev, load_pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        e, n = evaluate(model, img_dir, lab_dir, dev)
        print(f"{tag:<20}{np.median(e[:,0]):>10.2f}{np.percentile(e[:,0],95):>10.2f}"
              f"{e[:,0].max():>10.2f}{np.median(e[:,1]):>10.2f}{np.percentile(e[:,1],95):>10.2f}"
              f"{e[:,2].mean():>10.1f}{e[:,3].mean():>10.1f}")
        print(f"{'':<20}(共 {n} 张，误差单位 px)")


if __name__ == "__main__":
    main()
