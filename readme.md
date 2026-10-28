# AFD-Unet：PET 晶体 floodmap 光斑定位

用 U-Net 从单通道 floodmap 图像回归 36 个晶体光斑中心，输出 36 通道高斯热图，最终生成 Voronoi LUT 查找表。

---

## 1. 项目简介

### 任务

PET 探测器的 floodmap（泛场图）中，36 个晶体对应 36 个光斑。需要精确定位每个光斑中心，再按中心点生成 Voronoi 分区，作为后续事件归属的查找表（LUT）。

### 方法

| 项 | 内容 |
|---|---|
| 模型 | U-Net + ResNet18 编码器（ImageNet 预训练） |
| 输入 | 单通道 512×512 灰度图，归一化到 [0, 1] |
| 输出 | 36×512×512 热图，每通道对应一个光斑的高斯峰 |
| 标签 | 传统算法 `afd_pet_v3.py` 生成 (36,2) 坐标，转 sigma=4.0 高斯热图 |
| 损失 | `BCEWithLogitsLoss(pos_weight=50)` |
| 优化 | AdamW + linear warmup + CosineAnnealing |

### 为什么不用传统算法

`afd_pet_v3.py` 是原有的自适应光斑检测器（K-Means / 动态质心 + 质量评分 Q），在清晰图上可用，但在**模糊、粘连、低信噪比**的 floodmap 上需要反复调参，且逐图表现不一致。深度模型把这些先验学进权重，推理时一次前向完成。

传统算法仍保留两个用途：生成训练标签、以及在低置信度时兜底。

---

## 2. 快速开始

### 环境

```bash
pip install -r requirements.txt
```

实测通过：Python 3.13 / torch 2.14.1+cu126 / RTX 3060 12GB

### 数据准备

```bash
python convert_mat_to_npy.py      # .mat -> .npy
python generate_coords.py         # 传统算法生成标签
python merge_finetune_dataset.py  # 合并基础批次
python merge_blur_batch.py        # 合并模糊批次
python audit_test_leakage.py      # 入库前查重（必做）
```

### 训练

```bash
python train.py
```

训练配置在 `train.py` 顶部常量区：`NPY_DIR` / `COORD_DIR` / `LR` / `EPOCHS` / `RESUME_FROM` / `CKPT_NAME`。支持从已有权重继续微调。

### 推理

```bash
python batch_test_v1.py
```

批量输出「预测点 + Voronoi LUT」对比图。路径在文件顶部配置。

### 评估

```bash
python eval_position_error.py --data 1-2    # 定位精度（需 GT）
python compare_two_models.py                # 多模型置信度对比
python scan_conf_by_dataset.py              # 数据集置信度分布
python benchmark_python.py                  # 分阶段耗时基准
```

### ONNX 导出与 C++ 推理

```bash
# 1. 导出 ONNX（opset 14，输出 logits）
python export_onnx.py

# 2. 生成测试数据 + Python 参考基准
python gen_test_data.py --img test/floodmap-npy1-2/floodmap_10_1.npy

# 3. 构建并运行 C++
cd cpp_demo
cmake -S . -B build -G "MinGW Makefiles" && cmake --build build -j
./build/afd_demo.exe model/afd_unet.onnx test_data/floodmap_10_1.bin test_data/cpp

# 4. 与 Python 结果对比出图
cd ..
python verify_cpp_output.py --prefix floodmap_10_1
```

> `cpp_demo/third_party/`（ONNX Runtime，428MB）和 `cpp_demo/model/`（ONNX，55MB）已被 gitignore，首次使用需按 `cpp_demo/README.md` 下载并生成。

---

## 3. 数据集约定

### 命名规则

| 批次 | 文件名 | 说明 |
|---|---|---|
| 旧批次 | `floodmap_{i}_{j}.npy` | 无前缀 |
| 新批次 | `floodmap_260305_{i}_{j}.npy` | 260305 批次 |
| 模糊批次 | `floodmap_260305b_{i}_{j}.npy` | 1-7 组，b = blur |
| 标签 | 同名 + `_coords.npy` | shape (36,2)，顺序 = 通道顺序 |

`i` 是物理位置分组（1~32），`j` 是重复测量（1~6）。

通道顺序 = 3×3 模块（行优先）× 模块内 4 点（左上 / 右上 / 左下 / 右下），由 `afd_pet_v3.py` 的 `process()` 决定。

### 训练 / 测试划分

| 数据集 | 用途 | 张数 |
|---|---|---|
| `data/processed/finetune_*` | 训练 + 验证（按 i 分组 80/20） | 575 |
| `test/floodmap-npy1-2` | **Hold-out（模糊）** | 192 |
| `test/floodmap-npy_1_1_261020` | **Hold-out（清晰）** | 192 |
| `test/floodmap-npy_1_3_260812` | 训练集副本，禁止测试 | 192 |

验证集按 `i` 分组划分：同一个 `i` 的 18 张（旧 6 + 新 6 + 模糊 6）必须落在同一侧，否则近重复样本会跨集合泄漏，验证指标虚高。

### 入库前必做：查重

```bash
python audit_test_leakage.py
```

三层检查：字节级 md5 → 归一化内容哈希（容忍 dtype/量程差异）→ 标签 md5。

> 曾踩过坑：同事提供的一份数据与从服务器下载的是**同一批文件**，只是目录名不同，导致把训练集当成盲测集，白测了一轮。详见「坑四」。

---

## 4. 目录结构

```
AFD-Unet/
├── model.py                    # 模型定义（smp.Unet + ResNet18）
├── dataset.py                  # 数据集 + 在线热图生成 + 增强
├── afd_pet_v3.py               # 传统光斑检测器（标签来源 / 兜底）
├── train.py                    # 训练入口
├── batch_test_v1.py            # 批量推理 + Voronoi 出图
├── requirements.txt            # Python 依赖
│
├── export_onnx.py              # 导出 ONNX + PyTorch/ONNX 一致性验证
├── verify_channel_mapping.py   # 通道→LUT 编号映射等价性验证
├── gen_test_data.py            # 生成 .bin 测试数据 + Python 参考基准
├── verify_cpp_output.py        # C++ vs Python 输出对比 + 出图
├── benchmark_python.py         # Python 侧分阶段耗时基准
│
├── eval_position_error.py      # 定位精度评估（mean/max 误差）
├── compare_two_models.py       # 多模型置信度对比
├── scan_bad_labels.py          # 反查可疑标签
├── scan_conf_by_dataset.py     # 数据集置信度分布
├── audit_test_leakage.py       # 数据泄漏三层核查
│
├── convert_mat_to_npy.py       # .mat -> .npy
├── generate_coords.py          # 传统算法生成标签
├── merge_finetune_dataset.py   # 合并基础批次
├── merge_blur_batch.py         # 合并模糊批次
├── check_labels.py             # 人工审查（k 保留 / d 剔除）
│
├── visualize_36_channels.py    # 36 通道热图可视化
├── visualize_dataset.py        # 数据集抽样可视化
├── generate_final_lut.py       # 单图 LUT 演示
├── test_inference.py           # 单图推理可视化
├── overfit_test.py             # 10 张图过拟合测试
├── diag_*.py                   # 一次性诊断脚本
│
├── cpp_demo/                   # C++ 推理（ONNX Runtime）
│   ├── CMakeLists.txt
│   ├── include/afd_inference.h
│   ├── src/afd_inference.cpp   # 预处理/推理/ROI/通道重排/Voronoi
│   ├── src/main.cpp
│   ├── README.md
│   ├── model/                  # [gitignore] afd_unet.onnx
│   ├── test_data/              # [gitignore] .bin / .raw / .csv
│   └── third_party/            # [gitignore] ONNX Runtime 官方包
│
├── checkpoints/                # 权重
├── data/                       # processed/{npy_images, coords, finetune_*, bad_data}
├── logs/                       # 训练日志
├── res/                        # 诊断图
├── test/                       # 测试数据 + 结果图
└── weights/                    # 预训练权重
```

---

## 5. 训练记录

### V1：基线（BCE + pos_weight）

训练集 153，验证集 39。关闭平移和旋转增强（会导致边缘光斑出界），只保留噪点和对比度增强。

- 输入单通道 npy 矩阵，输出 36 通道热图
- 训练 150 轮，最佳 `Train Loss 0.00822 | Val Loss 0.00814`
- `BCEWithLogitsLoss(pos_weight=50)` + AdamW + CosineAnnealingLR

完整 150 轮日志见 `logs/v1.log`。

### V2：CoordConv 尝试（失败）

本想加入 CoordConv 让模型获得位置感知，但模型直接利用坐标画出高斯山包，彻底放弃图像特征，训练失败。

### V1.1：Focal Loss 尝试（失败）

换用 Focal Loss 后，模型发现"全图涂成 0.1 浅红"就能让 Loss 极低，预测 Max 仅 0.29，属于典型的摆烂解。

#### Focal Loss 的"盲区"

Focal Loss 公式里有一个 `(1-pt)^gamma`。它的初衷是：如果模型预测得很准（背景是 0，预测成 0.1），就把这个 Loss 降到接近 0。

#### 模型的"摆烂捷径"

图里背景占了 99.9%。模型发现，如果它把整张图都预测成 0.1 到 0.15 的浅红色：

- 对于海量背景，误差只有 0.1，Focal Loss 一算，惩罚微乎其微
- 对于真正的光斑，它预测成 0.29，虽然没到 1.0，但也没有偏离 0 太远，也没有受到毁灭性惩罚

综合下来，总 Loss 暴跌到了 0.0008。

### V1.2：Hybrid Loss 尝试（失败）

改混合损失函数，重点惩罚假峰，但把模型吓破胆，连真峰都不敢画。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, pos_weight=50.0, false_peak_penalty=10.0):
        super().__init__()
        # 1. V1 的成功基石：带权重的 BCE
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        # 2. 假峰惩罚系数
        self.false_peak_penalty = false_peak_penalty

    def forward(self, logits, targets):
        # 基础 BCE Loss (V1 的配方)
        loss_bce = self.bce(logits, targets)

        # 把 logits 转成概率 0~1
        probs = torch.sigmoid(logits)

        # ================= 核心：假峰惩罚 =================
        # 找出那些：标签是 0（纯背景），但模型预测值 > 0.05（产生了假峰）的像素
        # 我们希望把这些地方的预测值死死压到 0
        false_peak_mask = (targets == 0) & (probs > 0.05)

        if false_peak_mask.sum() > 0:
            # 对假峰区域算个均方误差，狠狠罚它
            loss_false = torch.mean(probs[false_peak_mask] ** 2)
        else:
            loss_false = 0.0

        # 总 Loss = 基础 BCE + 假峰惩罚
        return loss_bce + self.false_peak_penalty * loss_false
```

### 微调第 1 轮：新批次域适应（2026-10-22）

在 V1 权重基础上，加入 260305 新批次数据继续训练。

| 项 | 设置 |
|---|---|
| 数据 | 384 张（旧 192 + 新 192） |
| 起点 | `best_model_150_v1.pth` |
| LR | 3e-5（warmup 3 轮 + cosine） |
| 轮次 | 60（跑满） |
| 权重 | `best_model_finetune.pth` |

| 阶段 | Val | 旧批次 | 新批次 |
|---|---|---|---|
| Baseline | 0.01573 | 0.00819 | 0.02327 |
| Epoch 60 | **0.00544** | 0.00512 | 0.00576 |

新批次 loss 降 75.2%，旧批次降 37.5%（数据量翻倍带来的泛化增益），新旧差距从 2.84× 收窄到 1.13×。全程无过拟合，40 轮后基本收敛。

### 微调第 2 轮：模糊批次鲁棒性（2026-10-23）

加入 1-7 组的模糊 / 高噪点数据，专门提升低信噪比场景的鲁棒性。

| 项 | 设置 |
|---|---|
| 数据 | 575 张（384 + 模糊 192 - bad 1） |
| 起点 | `best_model_finetune.pth` |
| LR | 2e-5（warmup 2 轮 + cosine） |
| 轮次 | 25（跑满） |
| 权重 | `best_model_finetune_blur.pth` |

| 阶段 | Val | 旧批次 | 新批次 |
|---|---|---|---|
| Baseline | 0.00808 | 0.00924 | 0.00576 |
| Epoch 25 | **0.00516** | 0.00507 | 0.00535 |

在**独立的** 1-2 模糊集（192 张，与训练集零重叠）上验证定位精度：

| 指标 | 第 1 轮模型 | 第 2 轮模型 |
|---|---|---|
| mean 误差中位 | 6.48 px | **1.76 px** |
| mean 误差 p95 | 19.61 px | **4.97 px** |
| max 误差中位 | 35.55 px | **5.77 px** |
| 平均 >10px 点数/图 | 6.8 | **0.8** |
| 平均 >20px 点数/图 | 4.3 | **0** |

清晰图（261020）未退化：minConf 中位 0.984 → 0.988。说明模糊数据带来的是数据多样性，而不是分布偏移。

---

## 6. 踩坑记录

### 初衷：为什么要优化？

- 现象：V1（纯图像 + BCEWithLogitsLoss + pos_weight=50）虽然能精准找到光斑中心，但在其他光斑轮廓处，会残留亮度在 0.1~0.2 的微弱假峰
- 隐患：如果遇到极端干扰，argmax 可能会被假峰骗走

#### 坑一：引入 CoordConv（坐标硬编码）

- 操作：在输入端强行拼接 X、Y 坐标通道，试图让模型拥有"位置透视眼"
- 惨烈现象：Loss 瞬间暴跌（10 轮跌破 0.01），但预测图变成了马赛克色块，红叉直接飞到了画面边缘
- 踩坑原因：模型发现了作弊捷径。它根本不再看原图的光斑长什么样，而是直接背下了"在坐标 (0.5, 0.5) 画个点"的公式，彻底丧失了图像特征提取能力

#### 坑二：换用 Focal Loss（对付类别不平衡的"神药"）

- 操作：退回单通道输入，把 Loss 换成 Focal Loss（alpha=0.25, gamma=2.0）
- 惨烈现象：Loss 降到了极其离谱的 0.0008，但预测图变成了"满屏泛红"（全图 0.1~0.2 的红底），红叉彻底迷失
- 踩坑原因：Focal Loss 对背景（极易预测的像素）极度宽容。模型发现，只要把整张图都涂成 0.1 的浅红色，总 Loss 就会因为背景被忽略而变得极低。这就是典型的"均值陷阱"——模型摆烂了

#### 坑三：自研 Hybrid Loss（基础 BCE + 假峰重罚）

- 操作：退回 V1 的 BCE+pos_weight 配方，并强行加上"假峰惩罚项"（对背景区域预测值 >0.05 的像素乘 10 倍惩罚）
- 惨烈现象：初始 Loss 高达 3.33，然后死死卡在 0.11。跑完一看 36 宫格，全军覆没，预测图变成了一片纯黑
- 踩坑原因：模型被"10 倍的假峰惩罚"吓破了胆。它发现，想要绝对不犯错（不产生假峰），最好的办法就是"全图输出 0"。宁愿漏掉真光斑，也绝不冒一点产生假峰的风险。这是"极端逃避"

#### 坑四：数据重复（2026-10-23）

- 现象：`test/floodmap-npy_1_3_260812` 被当作盲测集，测出 191/192 的好成绩；后来发现它与训练集的 `data/processed/npy_images` **md5 完全相同**——一份是同事给的，一份是从服务器下载的，同一批文件换了目录名
- 后果：那一轮"盲测"实际上是训练集测试，成绩不可信
- 教训：**任何新数据入库前，先跑 `audit_test_leakage.py`**。只比文件名不够，必须比内容（md5 + 归一化内容哈希）

#### 坑五：编号排序（2026-10-23）

- 现象：某张图 36 个点定位误差平均只有 2.19px（最大 8.46px），但生成的 Voronoi LUT 明显错乱
- 根因：`batch_test_v1.py` 原来按 x 坐标排序、每 6 个切一列来生成 1~36 编号。floodmap 边缘两列的 x 只差 10~15px，单点抖动 3px 就会跨列越位，导致 **7/36 个编号错位**，其中 1 个跨列差 6 位
- 修复：编号改由通道索引直接推导，完全不看坐标

```python
m = ch // 4; i, j = m // 3, m % 3; k = ch % 4
row = i * 2 + (0 if k < 2 else 1)
col = j * 2 + (0 if k % 2 == 0 else 1)
rank = col * 6 + (6 - row)      # 列从左到右，行从下到上
```

- 教训：**结构化信息（通道序号）比几何推断（坐标排序）可靠**。能用索引就不要用坐标

---

## 7. 当前最佳配置

### 推荐权重

`checkpoints/best_model_finetune_blur.pth`（575 张训练，清晰 + 模糊兼顾）

### 关键指标

| 数据集 | 指标 | 值 |
|---|---|---|
| 1-2（独立模糊，192 张） | mean 误差中位 | 1.76 px |
| 1-2 | 平均 >20px 点数/图 | 0 |
| 261020（独立清晰，192 张） | minConf 中位 | 0.988 |
| 训练集（575 张） | mean 误差中位 | 1.17 px |

### 已知限制

1. **极暗 / 极粘连光斑**仍可能偏 10~20px（1-2 上平均每张 0.8 个点）
2. **置信度低 ≠ 定位错**：conf 0.5 的点位置可能仍然准确，只是峰弱。`batch_test_v1.py` 中 conf < 0.80 会打警告，建议人工复核而非直接丢弃
3. **标签质量是天花板**：GT 由传统算法生成，边缘光斑本身有 1~2px 误差，继续压低模型误差可能是在拟合标签噪声

---

## 8. ONNX 导出与 C++ 部署

### 流程

```
PyTorch (.pth) → ONNX (opset 14) → C++ (ONNX Runtime) → 36 点 + Voronoi LUT
```

导出脚本 `export_onnx.py`，C++ 实现见 `cpp_demo/`。

### 四层验证

| 层级 | 对比项 | 结果 |
|---|---|---|
| 1 | 通道映射公式 vs `batch_test_v1.py` | 36/36 完全等价 |
| 2 | ONNX vs PyTorch 输出 | max\|diff\| = 1.02e-4，**36 点 argmax 完全一致** |
| 3 | C++ vs Python 36 点坐标 | **0 误差**（模糊图 + 清晰图两张样本均如此） |
| 4 | C++ vs Python LUT | 0.047% ~ 0.065%（全在分区边界） |

第 2 层的 1e-4 是 FP32 深层网络累加顺序不同的正常误差（相对误差约 3e-6），不影响 argmax。

第 4 层的差异来自 `scipy.KDTree` 与暴力距离在"两点等距"时的取舍不同，全部落在晶体分区边界，不影响使用。

### 性能对比

单帧 512×512，10 次中位数：

| 实现 | 推理 | Voronoi | 合计 |
|---|---|---|---|
| Python PyTorch (CPU) | 126.37 ms | 58.54 ms | **185.49 ms** |
| Python ONNX Runtime | 68.91 ms | 58.54 ms | **128.03 ms** |
| C++ ONNX Runtime（自动线程） | 65.53 ms | 6.06 ms | **73.19 ms** |

- C++ 相比 PyTorch 提速 **2.5 倍**
- 加速几乎全部来自 Voronoi（**9.7 倍**）：`scipy.KDTree` 建树 + 查询开销大，C++ 暴力 940 万次距离在 `-O2` 下仅 6 ms
- 推理部分几乎无差异，因为同是 ORT 后端、同 CPU

**性能陷阱**：`SetIntraOpNumThreads(1)` 会让推理从 65 ms 变成 361 ms（5.5 倍）。默认值已设为 `0`（由 ORT 自动决定），且多线程与单线程结果完全一致。

### 关键坑

**ORT 1.31 移除了 Session 的路径重载**

```cpp
// 编译报错：no matching function
Ort::Session(env, model_path.c_str(), opts);
```

1.31 只剩 `Session(env, const Model&, opts)` 和 `Session(env, const void*, size_t, opts)`，路径版参数是 `const ORTCHAR_T*`（Windows 下为 `wchar_t*`）。解决方式是**读文件到内存再构造**，同时绕开路径编码问题。

**MinGW 没有 gendef，但不需要**

GNU ld 可以直接链接 ORT 的 MSVC dll（C ABI 兼容）。必须加 `-static-libgcc -static-libstdc++`，否则运行时缺 `libstdc++-6.dll`。

### 设计约定

第一版 C++ **严格复现 Python**，包括"低置信度只警告不处理"的行为。任何优化（低置信度兜底、buffer 复用、批处理）都应做成可选开关且默认关闭，否则无法与 Python 基准对齐。

### 仓库说明

以下目录体积大、可由脚本重建，已在 `.gitignore` 中排除：

| 路径 | 体积 | 重建方式 |
|---|---|---|
| `cpp_demo/third_party/` | 428 MB | 下载 ONNX Runtime 官方包（见 `cpp_demo/README.md`） |
| `cpp_demo/model/` | 55 MB | `python export_onnx.py` |
| `cpp_demo/build/` | 19 MB | `cmake --build build` |
| `cpp_demo/test_data/*.bin/*.raw/*.csv` | 5 MB | `python gen_test_data.py` |

