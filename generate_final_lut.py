import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from model import get_model
from dataset import FloodmapDataset
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
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

#拿验证集的图测试
# 1. 先实例化数据集对象，命名为 my_dataset (避免和模块重名)
my_dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=False)

# 2. 划分验证集索引
total_size = len(my_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
generator = torch.Generator().manual_seed(42)
_, val_idx = torch.utils.data.random_split(range(total_size), [train_size, val_size], generator=generator)

# 3. 拿到验证集中的第一张图
test_index = val_idx.indices[10]
print(f"本次测试使用的验证集文件索引: {test_index}")

# 4. 从这个实例化的对象里取数据
img_tensor, _ = my_dataset[test_index]  # 这里用 my_dataset，不用 dataset
img_input = img_tensor.unsqueeze(0).to(device)


# 预测
with torch.no_grad():
    pred_hm = torch.sigmoid(model(img_input)).cpu().squeeze(0).numpy()

raw_img = img_tensor[0].numpy()

# 1. 提取 36 个峰值点 (此时是模型通道的乱序)
points = []
for ch in range(36):
    y, x = np.unravel_index(np.argmax(pred_hm[ch]), (512, 512))
    points.append([x, y])
points = np.array(points)

# =========================================================
# 2. 核心修改：按图二的规则重新排序编号
# 规则：总共 6 列，从左到右排；每列内部 6 个点，从下往上排
# =========================================================
# 先按 X 坐标排序（从左到右）
points_sorted_x = points[np.argsort(points[:, 0])]

# 分成 6 列，每列 6 个点
cols = [points_sorted_x[i*6 : (i+1)*6] for i in range(6)]

reordered_points = []
for col in cols:
    # 每一列内部按 Y 坐标降序排列（从下到上，因为图像坐标 Y 轴朝下）
    col_sorted_y = col[np.argsort(col[:, 1])[::-1]]
    reordered_points.extend(col_sorted_y)

points = np.array(reordered_points) # 此时 points[0] 就是区域1，points[35] 就是区域36
# =========================================================

# 3. KDTree 生成 Voronoi LUT
grid_y, grid_x = np.mgrid[0:512, 0:512]
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
tree = KDTree(points)
_, indices = tree.query(grid_points)
lut_map = indices.reshape(512, 512) + 1  # 1~36 编号

# =========================================================
# 4. 生成 36 种区分度极高的颜色
# 使用 HSV 色轮均匀采样 36 种颜色，绝对不重复
# =========================================================
colors = plt.cm.hsv(np.linspace(0, 1, 36))
custom_cmap = mcolors.ListedColormap(colors)

# 5. 画出最终结果
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# 左边：原图 + 36个红叉
axes[0].imshow(raw_img, cmap='gray')
axes[0].scatter(points[:, 0], points[:, 1], c='red', s=40, marker='x')
axes[0].set_title("Predicted 36 Points")
axes[0].axis('off')

# 右边：完美的 Voronoi LUT
im = axes[1].imshow(lut_map, cmap=custom_cmap, interpolation='nearest', vmin=1, vmax=36)
axes[1].set_title("Final Voronoi LUT (Renumbered)")
# 标上编号
for i, (x, y) in enumerate(points):
    # 把文字改成白色，加黑色描边，确保在任何底色上都能看清
    axes[1].text(x, y, str(i+1), color='white', fontsize=10, fontweight='bold', 
                 ha='center', va='center', 
                 path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
axes[1].axis('off')

plt.tight_layout()
plt.show()