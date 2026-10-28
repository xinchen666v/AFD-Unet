# -*- coding: utf-8 -*-
"""
合并数据集：
  旧批次  data/processed/{npy_images,coords}                -> 保持原名
  新批次  test/floodmap-npy1-18 (图片)                      -> floodmap_260305_{i}_{j}.npy
          test/.../floodmap_1_18_output/data (标签 _pts.npy) -> floodmap_260305_{i}_{j}_coords.npy

目标：
  data/processed/finetune_npy_images/
  data/processed/finetune_coords/
"""
import os
import re
import sys
import shutil
from collections import Counter

BASE = r"D:\PyProject\AFD-Unet"

OLD_IMG_DIR = os.path.join(BASE, "data", "processed", "npy_images")
OLD_CRD_DIR = os.path.join(BASE, "data", "processed", "coords")
NEW_IMG_DIR = os.path.join(BASE, "test", "floodmap-npy1-18")
NEW_CRD_DIR = os.path.join(
    BASE, "test", "2-7-18结果", "260305floodmap", "floodmap",
    "floodmap_1_18_output", "data"
)

DST_IMG_DIR = os.path.join(BASE, "data", "processed", "finetune_npy_images")
DST_CRD_DIR = os.path.join(BASE, "data", "processed", "finetune_coords")

PREFIX = "260305"

RE_IMG = re.compile(r"^floodmap_(\d+)_(\d+)\.npy$")
RE_OLD_CRD = re.compile(r"^floodmap_(\d+)_(\d+)_coords\.npy$")
RE_NEW_CRD = re.compile(r"^floodmap_a(\d+)_b(\d+)_pts\.npy$")


def list_files(d, pattern):
    """返回 [(filename, regex_match), ...]，按 i,j 数值排序"""
    if not os.path.isdir(d):
        raise FileNotFoundError(f"目录不存在: {d}")
    out = []
    for f in os.listdir(d):
        m = pattern.match(f)
        if m:
            out.append((f, m))
    out.sort(key=lambda t: (int(t[1].group(1)), int(t[1].group(2))))
    return out


def copy_checked(src, dst):
    if not os.path.isfile(src):
        raise FileNotFoundError(f"源文件缺失: {src}")
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)


def main():
    print("=" * 60)
    print("开始合并 finetune 数据集")
    print("=" * 60)

    os.makedirs(DST_IMG_DIR, exist_ok=True)
    os.makedirs(DST_CRD_DIR, exist_ok=True)

    # ---------- 收集 ----------
    old_imgs = list_files(OLD_IMG_DIR, RE_IMG)
    old_crds = list_files(OLD_CRD_DIR, RE_OLD_CRD)
    new_imgs = list_files(NEW_IMG_DIR, RE_IMG)
    new_crds = list_files(NEW_CRD_DIR, RE_NEW_CRD)

    print(f"[收集] 旧图片 {len(old_imgs)}  旧标签 {len(old_crds)}")
    print(f"[收集] 新图片 {len(new_imgs)}  新标签 {len(new_crds)}")

    # ---------- 断言数量一致 ----------
    assert len(old_imgs) == len(old_crds), "旧数据图片与标签数量不一致"
    assert len(new_imgs) == len(new_crds), "新数据图片与标签数量不一致"

    # ---------- 编号集合必须完全对齐 ----------
    old_img_keys = {(int(m.group(1)), int(m.group(2))) for _, m in old_imgs}
    old_crd_keys = {(int(m.group(1)), int(m.group(2))) for _, m in old_crds}
    assert old_img_keys == old_crd_keys, (
        f"旧数据编号不对齐, 图片多出={old_img_keys - old_crd_keys}, 标签多出={old_crd_keys - old_img_keys}"
    )

    new_img_keys = {(int(m.group(1)), int(m.group(2))) for _, m in new_imgs}
    new_crd_keys = {(int(m.group(1)), int(m.group(2))) for _, m in new_crds}
    assert new_img_keys == new_crd_keys, (
        f"新数据编号不对齐, 图片多出={new_img_keys - new_crd_keys}, 标签多出={new_crd_keys - new_img_keys}"
    )

    print(f"[校验] 编号集合完全对齐, i={min(k[0] for k in old_img_keys)}~{max(k[0] for k in old_img_keys)}, "
          f"j={min(k[1] for k in old_img_keys)}~{max(k[1] for k in old_img_keys)}")

    # ---------- Step 1: 旧数据（原名） ----------
    n = 0
    for fname, _ in old_imgs:
        copy_checked(os.path.join(OLD_IMG_DIR, fname), os.path.join(DST_IMG_DIR, fname))
        n += 1
    print(f"[Step1] 旧图片复制完成: {n}")

    n = 0
    for fname, _ in old_crds:
        copy_checked(os.path.join(OLD_CRD_DIR, fname), os.path.join(DST_CRD_DIR, fname))
        n += 1
    print(f"[Step1] 旧标签复制完成: {n}")

    # ---------- Step 2: 新数据（改名加前缀） ----------
    n = 0
    for fname, m in new_imgs:
        i, j = int(m.group(1)), int(m.group(2))
        dst_name = f"floodmap_{PREFIX}_{i}_{j}.npy"
        copy_checked(os.path.join(NEW_IMG_DIR, fname), os.path.join(DST_IMG_DIR, dst_name))
        n += 1
    print(f"[Step2] 新图片复制+改名完成: {n}")

    n = 0
    for fname, m in new_crds:
        i, j = int(m.group(1)), int(m.group(2))
        dst_name = f"floodmap_{PREFIX}_{i}_{j}_coords.npy"
        copy_checked(os.path.join(NEW_CRD_DIR, fname), os.path.join(DST_CRD_DIR, dst_name))
        n += 1
    print(f"[Step2] 新标签复制+改名完成: {n}")

    # ---------- Step 3: 复核 ----------
    print("-" * 60)
    imgs = sorted([f for f in os.listdir(DST_IMG_DIR) if f.endswith(".npy")])
    crds = sorted([f for f in os.listdir(DST_CRD_DIR) if f.endswith(".npy")])
    print(f"[复核] finetune_npy_images: {len(imgs)} 个文件")
    print(f"[复核] finetune_coords    : {len(crds)} 个文件")

    # 配对检查：dataset.py 用 fname.replace('.npy', '_coords.npy')
    crd_set = set(crds)
    missing = [f for f in imgs if f.replace(".npy", "_coords.npy") not in crd_set]
    orphan = [f for f in crds if f.replace("_coords.npy", ".npy") not in set(imgs)]

    if missing:
        print(f"[ERROR] 有 {len(missing)} 张图片找不到对应标签，前5个: {missing[:5]}")
    else:
        print("[复核] 图片 -> 标签 配对率 100%")

    if orphan:
        print(f"[WARN] 有 {len(orphan)} 个标签没有对应图片，前5个: {orphan[:5]}")
    else:
        print("[复核] 无孤立标签")

    # 前缀统计
    cnt = Counter()
    for f in imgs:
        cnt["260305" if f.startswith(f"floodmap_{PREFIX}_") else "base"] += 1
    print(f"[复核] 来源分布: base={cnt['base']}, {PREFIX}={cnt['260305']}")

    # 抽样展示命名
    print("[复核] 命名示例:")
    for f in [x for x in imgs if not x.startswith(f"floodmap_{PREFIX}_")][:2]:
        print(f"    base : {f}  <->  {f.replace('.npy', '_coords.npy')}")
    for f in [x for x in imgs if x.startswith(f"floodmap_{PREFIX}_")][:2]:
        print(f"    {PREFIX}: {f}  <->  {f.replace('.npy', '_coords.npy')}")

    print("=" * 60)
    ok = (len(imgs) == len(old_imgs) + len(new_imgs)) and not missing and not orphan
    print("结果:", "✅ 全部通过" if ok else "❌ 存在问题，请检查上面日志")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
