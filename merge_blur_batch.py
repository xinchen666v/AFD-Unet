# -*- coding: utf-8 -*-
"""
把 1-7（模糊批次）并入 finetune 训练集
  图片: test/floodmap-npy1-7/floodmap_{i}_{j}.npy
  标签: test/2-7-18结果/.../floodmap_1_7_output/data/floodmap_a{i:02d}_b{j:02d}_pts.npy
  命名: floodmap_260305b_{i}_{j}.npy / _coords.npy
"""
import os
import re
import sys
import shutil

BASE = r"D:\PyProject\AFD-Unet"
NEW_IMG = os.path.join(BASE, "test", "floodmap-npy1-7")
NEW_CRD = os.path.join(
    BASE, "test", "2-7-18结果", "260305floodmap", "floodmap",
    "floodmap_1_7_output", "data"
)
DST_IMG = os.path.join(BASE, "data", "processed", "finetune_npy_images")
DST_CRD = os.path.join(BASE, "data", "processed", "finetune_coords")
PREFIX = "260305b"

RE_IMG = re.compile(r"^floodmap_(\d+)_(\d+)\.npy$")
RE_CRD = re.compile(r"^floodmap_a(\d+)_b(\d+)_pts\.npy$")


def collect(d, pattern):
    out = []
    for f in os.listdir(d):
        m = pattern.match(f)
        if m:
            out.append((f, int(m.group(1)), int(m.group(2))))
    return sorted(out, key=lambda t: (t[1], t[2]))


def copy_checked(src, dst):
    if not os.path.isfile(src):
        raise FileNotFoundError(f"源文件缺失: {src}")
    shutil.copy2(src, dst)


def main():
    imgs = collect(NEW_IMG, RE_IMG)
    crds = collect(NEW_CRD, RE_CRD)
    print(f"[收集] 1-7 图片 {len(imgs)}  标签 {len(crds)}")
    assert len(imgs) == len(crds) == 192, "数量异常"
    assert {(i, j) for _, i, j in imgs} == {(i, j) for _, i, j in crds}, "编号不对齐"

    n = 0
    for fname, i, j in imgs:
        copy_checked(os.path.join(NEW_IMG, fname),
                     os.path.join(DST_IMG, f"floodmap_{PREFIX}_{i}_{j}.npy"))
        n += 1
    print(f"[Step] 图片复制完成: {n}")

    n = 0
    for fname, i, j in crds:
        copy_checked(os.path.join(NEW_CRD, fname),
                     os.path.join(DST_CRD, f"floodmap_{PREFIX}_{i}_{j}_coords.npy"))
        n += 1
    print(f"[Step] 标签复制完成: {n}")

    imgs_all = [f for f in os.listdir(DST_IMG) if f.endswith(".npy")]
    crds_all = [f for f in os.listdir(DST_CRD) if f.endswith(".npy")]
    print(f"[复核] finetune_npy_images: {len(imgs_all)}")
    print(f"[复核] finetune_coords    : {len(crds_all)}")

    cs = set(crds_all)
    missing = [f for f in imgs_all if f.replace(".npy", "_coords.npy") not in cs]
    print(f"[复核] 配对缺失: {len(missing)}")
    assert not missing, f"存在未配对图片: {missing[:5]}"

    blur = sum(1 for f in imgs_all if f.startswith(f"floodmap_{PREFIX}_"))
    print(f"[复核] 来源分布: base={len(imgs_all)-blur-192 if False else len(imgs_all)-blur}, {PREFIX}={blur}")
    print("✅ 合并完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
