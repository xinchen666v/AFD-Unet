# -*- coding: utf-8 -*-
"""
Step 2: 导出 ONNX 并验证 PyTorch / ONNX 输出一致

输出: cpp_demo/model/afd_unet.onnx
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from model import get_model

CKPT = r"checkpoints/best_model_finetune_blur.pth"
OUT = r"cpp_demo/model/afd_unet.onnx"
SAMPLE = r"test/floodmap-npy1-2/floodmap_10_1.npy"

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    # 1. 构建模型并载入权重
    model = get_model(torch.device("cpu"), load_pretrained=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[1/4] 权重加载完成: {CKPT}")

    # 2. 用真实图做 dummy input（数值范围贴近实际）
    img = np.load(SAMPLE).astype(np.float32)
    img = img / (np.max(img) + 1e-8)
    dummy = torch.from_numpy(img)[None, None]      # [1,1,512,512]
    print(f"[2/4] dummy input: {tuple(dummy.shape)}, range [{img.min():.4f}, {img.max():.4f}]")

    # 3. 导出
    torch.onnx.export(
        model,
        dummy,
        OUT,
        input_names=["input"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"[3/4] 导出完成: {OUT}  ({size_mb:.1f} MB)")

    # 4. 验证 PyTorch vs ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    print("      inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("      outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    with torch.no_grad():
        y_pt = model(dummy).numpy()
    y_ort = sess.run(None, {"input": dummy.numpy()})[0]

    diff = np.abs(y_pt - y_ort)
    print(f"[4/4] 数值对比")
    print(f"      PyTorch: shape={y_pt.shape}, range=[{y_pt.min():.3f}, {y_pt.max():.3f}]")
    print(f"      ONNX   : shape={y_ort.shape}, range=[{y_ort.min():.3f}, {y_ort.max():.3f}]")
    print(f"      max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}")

    # 张量级阈值放宽到 1e-3：54MB 的深层网络 FP32 累加顺序不同，
    # 1e-4 量级属于正常误差（相对误差 ~3e-6）
    tensor_ok = diff.max() < 1e-3

    # 真正重要的判据：下游 ROI argmax 的 36 点坐标必须完全一致
    MS, R = 512 / 3.0, 85

    def extract_points(logits):
        hm = 1.0 / (1.0 + np.exp(-logits))          # sigmoid
        pts = np.zeros((36, 2), dtype=int)
        for ch in range(36):
            m = ch // 4
            i, j = m // 3, m % 3
            cx, cy = (j + 0.5) * MS, (i + 0.5) * MS
            x0, x1 = max(0, int(cx - R)), min(512, int(cx + R))
            y0, y1 = max(0, int(cy - R)), min(512, int(cy + R))
            ry, rx = np.unravel_index(np.argmax(hm[ch, y0:y1, x0:x1]), (y1 - y0, x1 - x0))
            pts[ch] = [x0 + rx, y0 + ry]
        return pts

    p_pt = extract_points(y_pt[0])
    p_ort = extract_points(y_ort[0])
    n_same = int((p_pt == p_ort).all(axis=1).sum())
    print(f"\n      ROI argmax 一致通道数: {n_same}/36")

    ok = tensor_ok and n_same == 36
    print(f"第 1 层验证: {'通过' if ok else '失败'}")
    print(f"  张量级 max|diff| < 1e-3 : {'OK' if tensor_ok else 'FAIL'}")
    print(f"  36 点坐标完全一致       : {'OK' if n_same == 36 else f'FAIL ({n_same}/36)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
