# -*- coding: utf-8 -*-
"""
Python 侧分阶段耗时基准（PyTorch vs ONNX Runtime，含后处理）

用法: python benchmark_python.py [--img path] [--n 10]
"""
import argparse
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import numpy as np
import torch
import onnxruntime as ort
from scipy.spatial import KDTree
from model import get_model

ONNX = r"cpp_demo/model/afd_unet.onnx"
CKPT = r"checkpoints/best_model_finetune_blur.pth"
MS, R = 512 / 3.0, 85


def bench(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000), float(np.min(ts) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=r"test/floodmap-npy_1_1_261020/floodmap_1_5.npy")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    img = np.load(args.img).astype(np.float32)
    n = args.n

    # ---- 模型加载耗时（一次性）----
    t0 = time.perf_counter()
    model = get_model(torch.device("cpu"), load_pretrained=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    t_load_pt = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    t_load_ort = (time.perf_counter() - t0) * 1000
    inp_name = sess.get_inputs()[0].name

    # ---- 预处理 ----
    def pre():
        x = img / (np.max(img) + 1e-8)
        return torch.from_numpy(x)[None, None]

    t_pre, _ = bench(pre, n)
    x = pre()

    # ---- PyTorch 推理 ----
    def pt_infer():
        with torch.no_grad():
            return model(x).numpy()

    t_pt, _ = bench(pt_infer, n)

    # ---- ORT 推理 ----
    xn = x.numpy()

    def ort_infer():
        return sess.run(None, {inp_name: xn})[0]

    t_ort, _ = bench(ort_infer, n)

    # ---- ROI 提取 ----
    logits = ort_infer()[0]
    hm = 1.0 / (1.0 + np.exp(-logits))

    def roi():
        pts = np.zeros((36, 2))
        for ch in range(36):
            m = ch // 4
            i, j = m // 3, m % 3
            cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
            x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
            y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
            ry, rx = np.unravel_index(np.argmax(hm[ch, y0:y1, x0:x1]), (y1 - y0, x1 - x0))
            pts[ch] = [x0 + rx, y0 + ry]
        return pts

    t_roi, _ = bench(roi, n)
    pts = roi()

    # ---- KDTree Voronoi ----
    def vor():
        gy, gx = np.mgrid[0:512, 0:512]
        tree = KDTree(pts)
        _, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        return idx

    t_vor, _ = bench(vor, n)

    print(f"样本: {os.path.basename(args.img)}   重复 {n} 次取中位数 (ms)")
    print("=" * 58)
    print(f"{'阶段':<26}{'耗时':>10}")
    print("-" * 58)
    print(f"{'模型加载 PyTorch':<26}{t_load_pt:>10.1f}")
    print(f"{'模型加载 ONNX Runtime':<26}{t_load_ort:>10.1f}")
    print(f"{'预处理 (归一化)':<26}{t_pre:>10.2f}")
    print(f"{'推理 PyTorch (CPU)':<26}{t_pt:>10.2f}")
    print(f"{'推理 ONNX Runtime (CPU)':<26}{t_ort:>10.2f}")
    print(f"{'ROI 提取':<26}{t_roi:>10.2f}")
    print(f"{'Voronoi (KDTree)':<26}{t_vor:>10.2f}")
    print("-" * 58)
    print(f"{'合计 PyTorch 管线':<26}{t_pre + t_pt + t_roi + t_vor:>10.2f}")
    print(f"{'合计 ORT 管线':<26}{t_pre + t_ort + t_roi + t_vor:>10.2f}")


if __name__ == "__main__":
    main()
