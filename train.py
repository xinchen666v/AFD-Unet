import os
# 强制 HuggingFace 离线，防止死等超时（必须在 import 其他库之前！）
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FloodmapDataset
from model import get_model

import re
import numpy as np

# ================= 配置 =================
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\finetune_npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\finetune_coords"
SAVE_DIR = r"D:\PyProject\AFD-Unet\checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 10
EPOCHS = 25            # 第二轮微调：新增 192 张模糊图，不需要跑太久
LR = 2e-5              # 已是第二轮微调，比上轮 3e-5 再降一档
WARMUP_EPOCHS = 2
PATIENCE = 8

# 第二轮微调起点：上一轮在 384 张上微调得到的权重
RESUME_FROM = r"D:\PyProject\AFD-Unet\checkpoints\best_model_finetune.pth"
CKPT_NAME = "best_model_finetune_blur.pth"   # 独立命名，保留上一轮权重

# ================= 分组划分工具 =================
GROUP_RE = re.compile(r"^floodmap_(?:(\d{6})\w*_)?(\d+)_(\d+)\.npy$")

def get_group_id(fname):
    """
    从文件名解析分组 id（物理位置 i）：
        floodmap_10_1.npy         -> 10
        floodmap_260305_10_1.npy  -> 10
    新旧批次中同一位置 i 归为同组，避免近重复样本跨 train/val。
    """
    m = GROUP_RE.match(fname)
    if m is None:
        raise ValueError(f"无法从文件名解析分组: {fname}")
    return int(m.group(2))


def evaluate(model, loader, criterion, device, src_tags=None):
    """
    评估并返回 (平均loss, 旧批次loss, 新批次loss)。
    src_tags: 与 loader 输出顺序一一对应的来源标记列表（"base"/"new"），
              用于分别观察模型对旧数据和新数据的适应情况。
    """
    model.eval()
    total = base_sum = new_sum = 0.0
    n_base = n_new = 0

    with torch.no_grad():
        for k, (images, heatmaps) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, heatmaps)

            v = loss.item()
            total += v
            if src_tags is not None:
                if src_tags[k] == "new":
                    new_sum += v
                    n_new += 1
                else:
                    base_sum += v
                    n_base += 1

    avg = total / max(1, len(loader))
    base = base_sum / n_base if n_base else float("nan")
    new = new_sum / n_new if n_new else float("nan")
    return avg, base, new


# class HybridLoss(nn.Module):
#     def __init__(self, pos_weight=50.0, false_peak_penalty=10.0):
#         super().__init__()
#         # 1. V1 的成功基石：带权重的 BCE
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
#         # 2. 假峰惩罚系数
#         self.false_peak_penalty = false_peak_penalty

#     def forward(self, logits, targets):
#         # 基础 BCE Loss (V1 的配方)
#         loss_bce = self.bce(logits, targets)
        
#         # 把 logits 转成概率 0~1
#         probs = torch.sigmoid(logits)
        
#         # ================= 核心：假峰惩罚 =================
#         # 找出那些：标签是 0（纯背景），但模型预测值 > 0.05（产生了假峰）的像素
#         # 我们希望把这些地方的预测值死死压到 0
#         false_peak_mask = (targets == 0) & (probs > 0.05)
        
#         if false_peak_mask.sum() > 0:
#             # 对假峰区域算个均方误差，狠狠罚它
#             loss_false = torch.mean(probs[false_peak_mask] ** 2)
#         else:
#             loss_false = 0.0
            
#         # 总 Loss = 基础 BCE + 假峰惩罚
#         return loss_bce + self.false_peak_penalty * loss_false





def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 划分训练集和验证集 (80% 训练, 20% 验证)
    full_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=True)
    val_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False) # 验证集不做增强
    
    # 按物理位置 i 分组划分：同一个 i 的所有图（旧批次 6 张 + 新批次 6 张）必须落在同一侧，
    # 否则同位置的近重复样本会同时进入 train 和 val，导致验证指标虚高。
    assert full_dataset.files == val_dataset.files, "两个 dataset 的文件顺序必须完全一致"

    group_ids = [get_group_id(f) for f in full_dataset.files]
    unique_groups = sorted(set(group_ids))

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(len(unique_groups))
    n_train_groups = int(round(0.8 * len(unique_groups)))

    train_groups = {unique_groups[k] for k in shuffled[:n_train_groups]}
    val_groups = {unique_groups[k] for k in shuffled[n_train_groups:]}

    train_idx = [idx for idx, g in enumerate(group_ids) if g in train_groups]
    val_idx = [idx for idx, g in enumerate(group_ids) if g in val_groups]

    assert not (set(train_idx) & set(val_idx)), "训练集与验证集存在索引重叠"

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset_subset = torch.utils.data.Subset(val_dataset, val_idx)

    print(f"分组划分: 共 {len(unique_groups)} 个 i 组 -> train {len(train_groups)} 组 / val {len(val_groups)} 组")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True,
        persistent_workers=True
        )
    val_loader = DataLoader(
        val_dataset_subset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
        )

    # 2. 初始化模型并加载微调起点
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        # 直接构建空模型再载入 v1，跳过无用的 ImageNet 权重下载（马上会被覆盖）
        model = get_model(device, load_pretrained=False)
        sd = torch.load(RESUME_FROM, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)   # 结构已实测一致
        print(f"[微调] 已加载起点权重: {RESUME_FROM}")
    else:
        model = get_model(device, load_pretrained=True)
        print(f"[微调] ⚠️ 未找到 {RESUME_FROM}，将从 ImageNet 预训练权重开始训练")

    # 损失函数
    pos_weight = torch.tensor([50.0]).to(device) # 如果训练效果不理想可以加大惩罚
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = HybridLoss(pos_weight=50.0, false_peak_penalty=10.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 学习率：先线性 warmup，再 cosine 衰减到 eta_min
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    scaler = torch.amp.GradScaler('cuda') # 修复警告

    # 验证集样本来源标记，用于分别统计旧/新批次 loss
    val_src = ["new" if full_dataset.files[i].startswith("floodmap_260305_") else "base"
               for i in val_idx]

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset_subset)}")

    # 3. 微调前 baseline
    # 历史 val loss 是在旧 192 张、旧随机划分上得到的，与当前 72 张分组验证集不可比，
    # 必须重新测一次，否则无法判断微调到底是变好还是变坏。
    base_loss, base_old, base_new = evaluate(model, val_loader, criterion, device, val_src)
    print(f"[Baseline] Val Loss={base_loss:.5f}  (旧批次 {base_old:.5f} | 新批次 {base_new:.5f})")

    best_val_loss = base_loss   # 以 baseline 为门槛，只有更优才保存
    patience_counter = 0

    print("开始微调...")

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, heatmaps in pbar:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()

            # 前向传播 (修复警告)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, heatmaps)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.5f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 4. 验证阶段
        avg_val_loss, val_old, val_new = evaluate(model, val_loader, criterion, device, val_src)
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1}/{EPOCHS}] lr={cur_lr:.2e} | Train {avg_train_loss:.5f} | "
              f"Val {avg_val_loss:.5f} (旧 {val_old:.5f} | 新 {val_new:.5f})")

        # 5. 保存最佳模型 + early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, CKPT_NAME))
            print(f"  -> Saved {CKPT_NAME} (Val Loss: {best_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停: 连续 {PATIENCE} 轮验证集无提升，停止训练")
                break

    # 6. 训练总结
    if best_val_loss < base_loss:
        print(f"微调完成: best Val Loss={best_val_loss:.5f} "
              f"(baseline {base_loss:.5f}, 提升 {(base_loss - best_val_loss) / base_loss * 100:.2f}%)")
        print(f"权重已保存: {os.path.join(SAVE_DIR, CKPT_NAME)}")
    else:
        print(f"⚠️ 本次微调未超过 baseline ({base_loss:.5f})，未保存新 checkpoint，建议沿用原权重")

if __name__ == "__main__":
    train()