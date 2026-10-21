import matplotlib.pyplot as plt
from dataset import FloodmapDataset

NPY_DIR = r"D:\PyProject\AFD-Unet\data\processed\npy_images"
COORD_DIR = r"D:\PyProject\AFD-Unet\data\processed\coords"

dataset = FloodmapDataset(NPY_DIR, COORD_DIR, is_train=True)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    fname = dataset.files[i]
    print(f"正在可视化文件: {fname}") # 终端打印文件名，方便你排查
    img, hm = dataset[i]
    
    # 画原图
    axes[i, 0].imshow(img[0].numpy(), cmap='gray')
    axes[i, 0].set_title(f"{fname} (Augmented)")
    axes[i, 0].axis('off')
    
    # 画热图 0
    axes[i, 1].imshow(hm[0].numpy(), cmap='hot')
    axes[i, 1].set_title(f"Ch0 (Max: {hm[0].max():.2f})")
    axes[i, 1].axis('off')
    
    # 画热图 16
    axes[i, 2].imshow(hm[16].numpy(), cmap='hot')
    axes[i, 2].set_title(f"Ch16 (Max: {hm[16].max():.2f})")
    axes[i, 2].axis('off')
    
plt.tight_layout()
plt.show()