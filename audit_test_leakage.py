# -*- coding: utf-8 -*-
"""
彻底核查 test/floodmap-npy1-2 是否以任何形式进入了训练集
三层检查：
  L1 字节级 md5
  L2 归一化内容哈希（排除 dtype/量程不同导致的 md5 差异）
  L3 标签 md5
"""
import os
import hashlib
import numpy as np
from tqdm import tqdm

TRAIN_IMG = r"data/processed/finetune_npy_images"
TRAIN_CRD = r"data/processed/finetune_coords"
TEST_IMG = r"test/floodmap-npy1-2"
TEST_CRD = r"test/2-7-18结果/260305floodmap/floodmap/floodmap_1_2_output/data"


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def content_hash(p):
    """归一化后取内容哈希：容忍 dtype / 量程差异"""
    a = np.load(p).astype(np.float32)
    a = a / (np.max(a) + 1e-8)
    return hashlib.md5(np.round(a, 4).tobytes()).hexdigest()


def build(d, fn):
    out = {}
    for f in os.listdir(d):
        if f.endswith(".npy"):
            out.setdefault(fn(os.path.join(d, f)), []).append(f)
    return out


def main():
    print("构建训练集指纹...")
    tr_md5 = build(TRAIN_IMG, md5_file)
    tr_cnt = build(TRAIN_IMG, content_hash)
    tr_crd = build(TRAIN_CRD, md5_file)
    print(f"  训练集: {sum(len(v) for v in tr_md5.values())} 张图片, "
          f"{len(tr_md5)} 个唯一 md5, {len(tr_cnt)} 个唯一内容哈希")

    tests = sorted([f for f in os.listdir(TEST_IMG) if f.endswith(".npy")])
    print(f"待核查 1-2: {len(tests)} 张\n")

    l1, l2, l3 = [], [], []
    for f in tqdm(tests, desc="核查中", leave=False):
        p = os.path.join(TEST_IMG, f)
        h1 = md5_file(p)
        h2 = content_hash(p)
        if h1 in tr_md5:
            l1.append((f, tr_md5[h1]))
        if h2 in tr_cnt:
            l2.append((f, tr_cnt[h2]))

        # 标签：1-2 的 a{i}_b{j} 对应训练集的哪个文件
        parts = f.replace(".npy", "").split("_")
        i, j = int(parts[1]), int(parts[2])
        lp = os.path.join(TEST_CRD, f"floodmap_a{i:02d}_b{j:02d}_pts.npy")
        if os.path.isfile(lp):
            h3 = md5_file(lp)
            if h3 in tr_crd:
                l3.append((f, tr_crd[h3]))

    print(f"\n{'='*66}")
    print(f"L1 字节级 md5 重复      : {len(l1)} 张")
    print(f"L2 归一化内容重复       : {len(l2)} 张")
    print(f"L3 标签 md5 重复        : {len(l3)} 张")
    print(f"{'='*66}")
    for tag, lst in [("L1", l1), ("L2", l2), ("L3", l3)]:
        if lst:
            print(f"{tag} 命中示例: {lst[:5]}")

    if not (l1 or l2 or l3):
        print("\n✅ 结论：1-2 的图片和标签均未以任何形式进入训练集")
    else:
        print("\n❌ 结论：存在重复，需进一步排查")

    # 同名文件内容差异抽查
    print("\n同名文件内容对比（抽查）:")
    for f in tests[:3]:
        trp = os.path.join(TRAIN_IMG, f)
        if os.path.isfile(trp):
            a = np.load(trp).astype(np.float32); b = np.load(os.path.join(TEST_IMG, f)).astype(np.float32)
            an = a / (np.max(a) + 1e-8); bn = b / (np.max(b) + 1e-8)
            print(f"  {f}: 训练集 max={a.max():.2f} vs 1-2 max={b.max():.2f} | "
                  f"归一化后最大差值={np.abs(an-bn).max():.4f}")


if __name__ == "__main__":
    main()
