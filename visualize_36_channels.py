import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_model
from dataset import FloodmapDataset

# 路径配置
NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
MODEL_PATH = r"D:\PyProject\AFD-Unet\checkpoints\best_model.pth"

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device, load_pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 获取验证集的图
my_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False)
total_size = len(my_dataset)
val_size = total_size - int(0.8 * total_size)
generator = torch.Generator().manual_seed(42)
_, val_idx = torch.utils.data.random_split(range(total_size), [total_size - val_size, val_size], generator=generator)

test_index = val_idx.indices[0]
img_tensor, _ = my_dataset[test_index]
img_input = img_tensor.unsqueeze(0).to(device)

# 预测
with torch.no_grad():
    pred_hm = torch.sigmoid(model(img_input)).cpu().squeeze(0).numpy()

raw_img = img_tensor[0].numpy()

# ==========================================
# 可视化 36 个通道的原始输出
# ==========================================
# 创建一个 6x6 的大图网格
fig, axes = plt.subplots(6, 6, figsize=(15, 15))
axes = axes.flatten() # 将二维的 axes 数组拉平，方便遍历

for ch in range(36):
    ax = axes[ch]
    # 画第 ch 个通道的热图，使用 'hot' 配色
    ax.imshow(pred_hm[ch], cmap='hot', vmin=0, vmax=1)
    ax.set_title(f"Channel {ch}", fontsize=10)
    ax.axis('off')

plt.suptitle(f"Raw 36-Channel Heatmap Output (Validation Index: {test_index})", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.95) # 给大标题留点空间
plt.show()