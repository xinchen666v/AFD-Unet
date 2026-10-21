import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_model
from dataset import FloodmapDataset
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制 HuggingFace 离线，拒绝联网

# 路径
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
MODEL_PATH = r"D:\PyProject\AFD-Unet\checkpoints\best_model.pth"

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = get_model(device)
model = get_model(device, load_pretrained=False) 
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # 切换到推理模式

# 随便抽一张图
dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False)
img_tensor, target_hm = dataset[0] # 拿第一张图
img_input = img_tensor.unsqueeze(0).to(device) # 增加 batch 维度

# 预测
with torch.no_grad():
    pred_hm = model(img_input)

# 转回 CPU 和 numpy
# pred_hm = pred_hm.cpu().squeeze(0).numpy()
pred_hm = torch.sigmoid(pred_hm).cpu().squeeze(0).numpy() # 加 Sigmoid 变成 0~1 概率
target_hm = target_hm.numpy()
raw_img = img_tensor[0].numpy()

# 可视化对比
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 原图
axes[0, 0].imshow(raw_img, cmap='gray')
axes[0, 0].set_title("Input Image")
axes[0, 0].axis('off')

# 2. 真实标签 通道0 (Target)
axes[0, 1].imshow(target_hm[0], cmap='hot', vmin=0, vmax=1)
axes[0, 1].set_title("Target Heatmap (Ch 0)")
axes[0, 1].axis('off')

# 3. 预测输出 通道0 (Prediction)
# 注意：这里我们先不加 vmin/vmax，看看模型真实的输出范围
im = axes[0, 2].imshow(pred_hm[0], cmap='hot')
axes[0, 2].set_title(f"Pred Heatmap (Ch 0)\nMax: {pred_hm[0].max():.4f}")
axes[0, 2].axis('off')
plt.colorbar(im, ax=axes[0, 2])


# 4. 真实标签 通道16 (Target)
axes[1, 0].imshow(target_hm[16], cmap='hot', vmin=0, vmax=1)
axes[1, 0].set_title("Target Heatmap (Ch 16)")
axes[1, 0].axis('off')

# 5. 预测输出 通道16 (Prediction)
im2 = axes[1, 1].imshow(pred_hm[16], cmap='hot')
axes[1, 1].set_title(f"Pred Heatmap (Ch 16)\nMax: {pred_hm[16].max():.4f}")
axes[1, 1].axis('off')
plt.colorbar(im2, ax=axes[1, 1])

# 6. 叠加找点
y_pred, x_pred = np.unravel_index(np.argmax(pred_hm[16]), (512, 512))
axes[1, 2].imshow(raw_img, cmap='gray')
axes[1, 2].scatter(x_pred, y_pred, c='red', s=50, marker='x')
axes[1, 2].set_title("Predicted Point on Image")
axes[1, 2].axis('off')

print(f"Target Max: {target_hm.max():.4f}, Min: {target_hm.min():.4f}")
print(f"Pred Max (After Sigmoid): {pred_hm.max():.4f}, Min: {pred_hm.min():.4f}")

plt.tight_layout()
plt.show()