import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_model

# ================= 路径配置 =================
# 指定你要看的那张图
# IMG_PATH = r"D:\PyProject\AFD-Unet\test\floodmap-npy_1_3_260812\floodmap_28_2.npy"
IMG_PATH = r"D:\PyProject\AFD-Unet\data\processed\npy_images\floodmap_28_2.npy"
# 你的微调后模型路径
MODEL_PATH = r"D:\PyProject\AFD-Unet\checkpoints\best_model_finetune.pth" 

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device, load_pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ================= 读取并处理单张图片 =================
# 1. 读取 .npy
img = np.load(IMG_PATH).astype(np.float32)
# 2. 归一化 (必须和训练时一样)
img = img / (np.max(img) + 1e-8)
# 3. 转为 Tensor 并加上 Batch 维度 [1, 1, 512, 512]
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

# ================= 预测 =================
with torch.no_grad():
    pred_hm = torch.sigmoid(model(img_tensor)).cpu().squeeze(0).numpy()

# ================= 可视化 36 个通道的原始输出 =================
fig, axes = plt.subplots(6, 6, figsize=(15, 15))
axes = axes.flatten() # 将二维的 axes 数组拉平

for ch in range(36):
    ax = axes[ch]
    # 画第 ch 个通道的热图
    ax.imshow(pred_hm[ch], cmap='hot', vmin=0, vmax=1)
    
    # 提取当前通道的最大值作为该点的置信度，写在标题上
    max_val = np.max(pred_hm[ch])
    ax.set_title(f"Ch {ch} (Max: {max_val:.2f})", fontsize=10)
    ax.axis('off')

# 提取文件名用于大标题
file_name = os.path.basename(IMG_PATH)
plt.suptitle(f"Raw 36-Channel Heatmap Output for: {file_name}", fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.93) # 给大标题留点空间
plt.show()

# # ================= 可视化：把预测点画在原图上 =================
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# # 拉伸对比度显示原图
# vmax = np.percentile(img, 99.5)
# if vmax == 0: vmax = 1.0
# ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)

# # 提取 36 个通道的最大值坐标
# points = []
# for ch in range(36):
#     y, x = np.unravel_index(np.argmax(pred_hm[ch]), (512, 512))
#     points.append([x, y])

# points = np.array(points)

# # 画红叉
# file_name = os.path.basename(IMG_PATH)
# ax.scatter(points[:, 0], points[:, 1], c='red', s=40, marker='x', linewidths=1.5)
# ax.set_title(f"Predicted 36 Points on: {file_name}")
# ax.axis('off')
# plt.show()




# import os
# os.environ["HF_HUB_OFFLINE"] = "1"
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from model import get_model
# from dataset import FloodmapDataset

# # 路径配置
# NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
# COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"
# MODEL_PATH = r"D:\PyProject\AFD-Unet\checkpoints\best_model_61_v1_2.pth"

# # 加载模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = get_model(device, load_pretrained=False)
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()

# # 获取验证集的图
# my_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False)
# total_size = len(my_dataset)
# val_size = total_size - int(0.8 * total_size)
# generator = torch.Generator().manual_seed(42)
# _, val_idx = torch.utils.data.random_split(range(total_size), [total_size - val_size, val_size], generator=generator)

# test_index = val_idx.indices[0]
# img_tensor, _ = my_dataset[test_index]
# img_input = img_tensor.unsqueeze(0).to(device)

# # 预测
# with torch.no_grad():
#     pred_hm = torch.sigmoid(model(img_input)).cpu().squeeze(0).numpy()

# raw_img = img_tensor[0].numpy()

# # ==========================================
# # 可视化 36 个通道的原始输出
# # ==========================================
# # 创建一个 6x6 的大图网格
# fig, axes = plt.subplots(6, 6, figsize=(15, 15))
# axes = axes.flatten() # 将二维的 axes 数组拉平，方便遍历

# for ch in range(36):
#     ax = axes[ch]
#     # 画第 ch 个通道的热图，使用 'hot' 配色
#     ax.imshow(pred_hm[ch], cmap='hot', vmin=0, vmax=1)
#     ax.set_title(f"Channel {ch}", fontsize=10)
#     ax.axis('off')

# plt.suptitle(f"Raw 36-Channel Heatmap Output (Validation Index: {test_index})", fontsize=20)
# plt.tight_layout()
# plt.subplots_adjust(top=0.95) # 给大标题留点空间
# plt.show()