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

# ================= 配置 =================
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
SAVE_DIR = r"D:\PyProject\AFD-Unet\checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 4       
EPOCHS = 150          # 快速验证，如果效果好，后面改成 150
LR = 1e-4

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 划分训练集和验证集 (80% 训练, 20% 验证)
    full_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=True)
    val_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False) # 验证集不做增强
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 修复随机重叠问题：使用固定的随机种子进行划分
    generator = torch.Generator().manual_seed(42) 
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    _, val_dataset_subset = torch.utils.data.random_split(val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset_subset, batch_size=1, shuffle=False, num_workers=0)

    # 2. 初始化模型
    # 注意：确保你的 model.py 里 get_model 能读取本地 weights 或者传入 None
    model = get_model(device, load_pretrained=True) 
    
    # 损失函数
    pos_weight = torch.tensor([50.0]).to(device) # 如果训练效果不理想可以加大惩罚
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    scaler = torch.amp.GradScaler('cuda') # 修复警告

    best_val_loss = float('inf')

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset_subset)}")
    print("开始训练...")

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
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, heatmaps)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # 5. 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  -> Saved best model! (Val Loss: {best_val_loss:.5f})")

if __name__ == "__main__":
    train()