# -*- coding: utf-8 -*-
"""
Step 1: 通道映射等价性验证

目的：证明"需求文档给出的通道->LUT编号公式"与 "batch_test_v1.py 现有实现" 完全等价。
       C++ 侧会照公式实现，若两者有偏差，C++ 结果将与 Python 对不上。

对比两条路径：
  A. batch_test_v1.py 的 order 数组（rank-1 -> ch）
  B. 需求公式（ch -> target_index）
"""
import numpy as np


def order_from_batch_test():
    """batch_test_v1.py:71-78 的现有实现"""
    order = np.zeros(36, dtype=int)   # order[rank-1] = ch
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        k = ch % 4
        row = i * 2 + (0 if k < 2 else 1)
        col = j * 2 + (0 if k % 2 == 0 else 1)
        order[col * 6 + (6 - row) - 1] = ch
    return order


def target_index_from_spec(ch):
    """需求文档给出的公式（返回 0-based index）"""
    m = ch // 4          # 大模块 0~8
    i = m // 3           # 模块行 0~2
    j = m % 3            # 模块列 0~2
    k = ch % 4           # 模块内 4 点：0 左上, 1 右上, 2 左下, 3 右下
    row = i * 2 + (1 if k >= 2 else 0)
    col = j * 2 + (1 if k % 2 == 1 else 0)
    return col * 6 + (6 - row) - 1


def main():
    order = order_from_batch_test()

    # 由公式反推 order：order[target_index(ch)] = ch
    order_from_spec = np.full(36, -1, dtype=int)
    for ch in range(36):
        ti = target_index_from_spec(ch)
        assert 0 <= ti < 36, f"ch{ch} 的 target_index 越界: {ti}"
        assert order_from_spec[ti] == -1, f"target_index {ti} 被重复占用"
        order_from_spec[ti] = ch

    same = np.array_equal(order, order_from_spec)
    print("=" * 64)
    print(f"{'ch':>3} {'模块(i,j)':>10} {'k':>3} {'row':>4} {'col':>4} {'LUT编号':>8}   {'order验证':>10}")
    print("-" * 64)
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        k = ch % 4
        row = i * 2 + (1 if k >= 2 else 0)
        col = j * 2 + (1 if k % 2 == 1 else 0)
        lut_id = col * 6 + (6 - row)
        ok = "OK" if order[lut_id - 1] == ch else "MISMATCH"
        print(f"{ch:>3} {'(%d,%d)' % (i, j):>10} {k:>3} {row:>4} {col:>4} {lut_id:>8}   {ok:>10}")

    print("=" * 64)
    print(f"A(batch_test) : {order.tolist()}")
    print(f"B(spec)       : {order_from_spec.tolist()}")
    print(f"\n两条路径完全等价: {'是' if same else '否'}")

    # 6x6 物理布局（row=0 为图像顶部，col=0 为图像左侧）
    grid = np.zeros((6, 6), dtype=int)
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        k = ch % 4
        row = i * 2 + (1 if k >= 2 else 0)
        col = j * 2 + (1 if k % 2 == 1 else 0)
        grid[row, col] = col * 6 + (6 - row)
    print("\n6x6 物理布局（row=0 为图像顶部，col=0 为左侧）:")
    print("       col0 col1 col2 col3 col4 col5")
    for r in range(6):
        print(f"  row{r}  " + " ".join(f"{grid[r, c]:>4}" for c in range(6)))

    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
