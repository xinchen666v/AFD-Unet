# AFD-Unet C++ 推理 Demo

把 PyTorch U-Net（`best_model_finetune_blur.pth`）导出为 ONNX，用 ONNX Runtime 在 C++ 中完成推理 + 后处理，输出 36 个光斑坐标和 512×512 的 Voronoi LUT。

## 目录结构

```
cpp_demo/
├── CMakeLists.txt
├── include/afd_inference.h        # 推理类接口
├── src/afd_inference.cpp          # 核心实现（预处理/推理/ROI/Voronoi）
├── src/main.cpp                   # 命令行入口
├── model/afd_unet.onnx            # 由 ../export_onnx.py 生成
├── test_data/
│   ├── floodmap_10_1.bin          # 512×512 float32 原始数据（未归一化）
│   ├── ref_points.csv             # Python 参考基准
│   ├── ref_lut.raw                # Python 参考 LUT
│   ├── cpp_points.csv             # C++ 输出
│   ├── cpp_lut.raw                # C++ 输出
│   └── compare.png                # 对比图
├── build/                         # 构建产物
└── third_party/
    └── onnxruntime-win-x64-1.31.1/
        ├── include/
        └── lib/onnxruntime.dll
```

## 构建

### 方式 1：g++ 命令行（最快）

```bash
cd cpp_demo
g++ -std=c++17 -O2 -static-libgcc -static-libstdc++ \
    -Iinclude -Ithird_party/onnxruntime-win-x64-1.31.1/include \
    src/main.cpp src/afd_inference.cpp \
    third_party/onnxruntime-win-x64-1.31.1/lib/onnxruntime.dll \
    -o build/afd_demo.exe

cp third_party/onnxruntime-win-x64-1.31.1/lib/onnxruntime.dll build/
```

### 方式 2：CMake

```bash
cd cpp_demo
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build -j
```

CLion 直接打开 `cpp_demo/CMakeLists.txt` 即可。

> MinGW 没有 `gendef` 也能用——GNU ld 可以直接链接 ORT 的 MSVC dll（C ABI 兼容）。
> 必须加 `-static-libgcc -static-libstdc++`，否则运行时缺 `libstdc++-6.dll`。

## 运行

```bash
./build/afd_demo.exe model/afd_unet.onnx test_data/floodmap_10_1.bin test_data/cpp
```

参数：`<onnx> <bin> <输出前缀>`，三个都可省略（有默认值）。

输出：
- `<前缀>_points.csv` — 36 点，字段 `index,x,y,conf`，按 LUT 编号 1~36 排序
- `<前缀>_lut.raw` — 512×512 uint8 裸二进制，值 1~36，行优先

## 输入格式

- 512×512 float32，行优先，无 header
- **原始未归一化数据**（与 `.npy` 内容一致），归一化在 C++ 内部完成：`img / (max + 1e-8)`

## 后处理要点

| 步骤 | 说明 |
|---|---|
| 归一化 | 全局 max 归一化，`max` 由输入数据现算 |
| 推理 | 输出 logits（模型 `activation=None`），**不是概率** |
| ROI 搜索 | 每通道在 3×3 模块内搜峰，`radius = 85`（模块半宽） |
| 置信度 | **只对 ROI 最大值做 sigmoid**，避免整图 940 万次 exp |
| 通道重排 | 用通道索引直接算 LUT 编号，**不依赖坐标排序**（见下） |
| Voronoi | 暴力遍历 262144 像素 × 36 点，替代 `scipy.KDTree` |

### 通道 → LUT 编号

```
m = ch / 4;  i = m / 3;  j = m % 3;  k = ch % 4
row = i*2 + (k >= 2 ? 1 : 0)
col = j*2 + (k % 2 == 1 ? 1 : 0)
lut_id = col*6 + (6 - row)          // 1..36
```

**不要用 x 坐标排序分列**——floodmap 边缘两列 x 仅差 10~15px，单点抖动 3px 就会跨列错号（实测 7/36 个编号错位）。

## 验证结果

| 层级 | 对比项 | 结果 |
|---|---|---|
| 1 | 通道映射公式 vs `batch_test_v1.py` | 36/36 完全等价 |
| 2 | ONNX vs PyTorch 输出 | max\|diff\| = 1.02e-4，**36 点 argmax 完全一致** |
| 3 | C++ vs Python 36 点坐标 | **0 误差** |
| 4 | C++ vs Python LUT | 169 / 262144 = **0.0645%** |

LUT 的 169 个差异像素全部位于分区边界——`scipy.KDTree` 与暴力距离在"两点等距"时的取舍不同，不影响视觉和使用。

性能：单帧 **375 ms**（RTX 3060 上含模型加载、推理、Voronoi；`-O2` 编译，单线程）。

## 与 Python 版本的一致性约束

第一版 C++ **严格复现** Python 逻辑，包括"低置信度只警告不处理"的行为。任何优化（低置信度兜底、buffer 复用、多线程）都应做成**可选开关且默认关闭**，否则无法与 Python 基准对齐。
