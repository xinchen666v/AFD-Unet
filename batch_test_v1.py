import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from scipy.spatial import KDTree
from tqdm import tqdm
from model import get_model

# ================= 路径配置（请修改为你自己的） =================
# 1. 你的全新 Floodmap 文件夹（里面全是 .npy 文件）
NEW_DATA_DIR = r"D:\PyProject\AFD-Unet\test\floodmap-npy_1_1_261020" 
# 2. 结果图片保存的文件夹
SAVE_DIR = r"D:\PyProject\AFD-Unet\test\test_results_1_1_261020_finetune_blur"
# 3. 你的 V1 最佳权重
MODEL_PATH = r"D:\PyProject\AFD-Unet\checkpoints\best_model_finetune_blur.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 加载模型 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device, load_pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ================= 核心处理函数 =================
def process_single_image(img_path, save_name):
    # 1. 读取并归一化
    img = np.load(img_path).astype(np.float32)
    img = img / (np.max(img) + 1e-8)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 512, 512]

    # 2. 预测
    with torch.no_grad():
        pred_hm = torch.sigmoid(model(img_tensor)).cpu().squeeze(0).numpy()

    # 3. 带 ROI 限制的局部搜索 (核心防串扰机制)
    points = []
    img_size = 512
    module_size = img_size / 3.0
    confidences = []

    for ch in range(36):
        m_idx = ch // 4
        m_y, m_x = m_idx // 3, m_idx % 3
        
        center_x, center_y = (m_x + 0.5) * module_size, (m_y + 0.5) * module_size
        
        radius = 85  # 搜索半径 = 3x3 模块半宽(512/3/2)，避免跨模块抢峰
        x_min, x_max = max(0, int(center_x - radius)), min(img_size, int(center_x + radius))
        y_min, y_max = max(0, int(center_y - radius)), min(img_size, int(center_y + radius))
        
        roi_pred = pred_hm[ch, y_min:y_max, x_min:x_max]
        
        # 找最大值及其置信度
        roi_y, roi_x = np.unravel_index(np.argmax(roi_pred), roi_pred.shape)
        confidences.append(np.max(roi_pred))
        
        global_x, global_y = x_min + roi_x, y_min + roi_y
        points.append([global_x, global_y])

    points = np.array(points)

    # 4. 置信度预警 (低于 0.80 说明该通道可能漏检/暗斑，建议人工复核)
    min_conf = np.min(confidences)
    if min_conf < 0.80:
        print(f"  ⚠️ 警告: {os.path.basename(img_path)} 最低通道置信度 {min_conf:.2f}，建议复核")

    # 5. 生成编号顺序：直接由通道索引推导，不再依赖坐标排序
    #    标签顺序 = 3x3 模块(行i, 列j) × 模块内 4 点(左上/右上/左下/右下)
    #    映射到 6x6 网格：row = i*2 + (k>=2),  col = j*2 + (k%2)
    #    Voronoi 编号：col*6 + (6-row)，列从左到右、行从下到上
    #    注意：用坐标排序时，边缘两列的 x 仅差 10~15px，单点抖动就会跨列错号
    #          （实测该错误在 t1 样本上影响 7/36 个编号，其中 1 个跨列差 6 位）
    order = np.zeros(36, dtype=int)   # order[rank-1] = 通道号
    for ch in range(36):
        m = ch // 4
        i, j = m // 3, m % 3
        k = ch % 4
        row = i * 2 + (0 if k < 2 else 1)
        col = j * 2 + (0 if k % 2 == 0 else 1)
        order[col * 6 + (6 - row) - 1] = ch
    points = points[order]

    # 6. KDTree 生成 Voronoi LUT
    grid_y, grid_x = np.mgrid[0:512, 0:512]
    tree = KDTree(points)
    _, indices = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    lut_map = indices.reshape(512, 512) + 1

    # 7. 画图与保存
    colors = plt.cm.hsv(np.linspace(0, 1, 36))
    custom_cmap = mcolors.ListedColormap(colors)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmax = np.percentile(img, 99.5)
    if vmax == 0: vmax = 1.0

    # 左图
    im0 = axes[0].imshow(img, cmap='gray', vmin=0, vmax=vmax)
    axes[0].scatter(points[:, 0], points[:, 1], c='red', s=40, marker='x', linewidths=1.5)
    axes[0].set_title(f"Predicted 36 Points (Min Conf: {min_conf:.2f})", fontsize=14)
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 右图
    axes[1].imshow(lut_map, cmap=custom_cmap, interpolation='nearest', vmin=1, vmax=36)
    axes[1].set_title("Final Voronoi LUT", fontsize=14)
    for i, (x, y) in enumerate(points):
        axes[1].text(x, y, str(i+1), color='white', fontsize=10, fontweight='bold', 
                     ha='center', va='center', 
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig) # 极其重要：清理内存，防止画图太多卡死

# ================= 批量循环 =================
if __name__ == "__main__":
    all_files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith('.npy')]
    print(f"共找到 {len(all_files)} 张测试图，开始批量盲测...")
    
    for fname in tqdm(all_files):
        img_path = os.path.join(NEW_DATA_DIR, fname)
        save_name = fname.replace('.npy', '_result.png')
        try:
            process_single_image(img_path, save_name)
        except Exception as e:
            print(f"\n❌ 处理 {fname} 时出错: {e}")

    print(f"\n🎉 所有测试完成！结果保存在: {SAVE_DIR}")