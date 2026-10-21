import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import FloodmapDataset
from model import get_model
from torch.utils.data import DataLoader, Subset

# ================= 配置 =================
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"

# 关键：is_train=False 强制关闭数据增强！
full_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False, sigma=3.0)

# 只取前 10 张图做测试
overfit_dataset = Subset(full_dataset, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
overfit_loader = DataLoader(overfit_dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型 (使用本地权重，避免联网)
model = get_model(device, load_pretrained=True)
# 如果模型输出层有限制，确保没有 sigmoid 激活
if hasattr(model, 'segmentation_head'):
    model.segmentation_head[0].activation = nn.Identity()

# 损失函数和优化器
pos_weight = torch.tensor([100.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("开始过拟合测试 (只训练10张图)...")
epochs = 150
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for images, heatmaps in overfit_loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss / len(overfit_loader):.5f}")

# ================= 测试推理结果 =================
print("\n训练结束，生成可视化图...")
model.eval()
# 拿第 0 张图测试
img_tensor, target_hm = full_dataset[0]
img_input = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda'):
        pred_logits = model(img_input)
        # 必须手动套一个 Sigmoid 才能变成 0~1 概率图
        pred_hm = torch.sigmoid(pred_logits).cpu().squeeze(0).numpy()

target_hm = target_hm.numpy()
raw_img = img_tensor[0].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(raw_img, cmap='gray')
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(target_hm[0], cmap='hot', vmin=0, vmax=1)
axes[1].set_title("Target Heatmap (Ch 0)")
axes[1].axis('off')

im = axes[2].imshow(pred_hm[0], cmap='hot', vmin=0, vmax=1)
axes[2].set_title(f"Pred Heatmap (Ch 0)\nMax: {pred_hm[0].max():.4f}")
axes[2].axis('off')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()