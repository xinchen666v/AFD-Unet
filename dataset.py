# #V2 加入CoordConv，让模型拥有空间感知能力
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import albumentations as A

# class FloodmapDataset(Dataset):
#     def __init__(self, img_dir, coord_dir, img_size=512, sigma=4.0, is_train=True):
#         # ... (前面的初始化保持不变) ...
#         self.img_dir = img_dir
#         self.coord_dir = coord_dir
#         self.img_size = img_size
#         self.sigma = sigma
#         self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        
#         if is_train:
#             self.transform = A.Compose([
#                 A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
#                 A.GaussianBlur(blur_limit=(3, 5), p=0.2),
#                 A.GaussNoise(std_range=(0.01, 0.03), p=0.2), 
#             ])
#         else:
#             self.transform = None

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         img_path = os.path.join(self.img_dir, fname)
#         coord_path = os.path.join(self.coord_dir, fname.replace('.npy', '_coords.npy'))

#         img = np.load(img_path).astype(np.float32)
#         coords = np.load(coord_path)
        
#         coords = np.clip(coords, 0, self.img_size - 1)
#         coords = np.nan_to_num(coords, nan=0.0)
#         img = img / (np.max(img) + 1e-8)

#         # 1. 增强（只作用于原图 [H, W]）
#         if self.transform:
#             augmented = self.transform(image=img)
#             img = augmented['image']

#         # 2. 生成 36 通道高斯热图 (保持原样)
#         heatmap = np.zeros((36, self.img_size, self.img_size), dtype=np.float32)
#         yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
#         for ch in range(36):
#             x, y = coords[ch]
#             heatmap[ch] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))

#         # # ================= CoordConv 核心改造 =================
#         # # 3. 生成归一化的坐标通道 [0, 1]
#         # x_coord = np.linspace(0, 1, self.img_size, dtype=np.float32)
#         # y_coord = np.linspace(0, 1, self.img_size, dtype=np.float32)
#         # xx_coord, yy_coord = np.meshgrid(x_coord, y_coord)
        
#         # # 4. 拼接成 3 通道: [灰度, X坐标, Y坐标]
#         # img_3ch = np.stack([img, xx_coord, yy_coord], axis=0) # 形状 [3, 512, 512]
#         # # ======================================================

#         # img_tensor = torch.from_numpy(img_3ch)
#         img_tensor = torch.from_numpy(img).unsqueeze(0) # 恢复单通道 [1, 512, 512]
#         heatmap_tensor = torch.from_numpy(heatmap)

#         return img_tensor, heatmap_tensor

# V1版本
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class FloodmapDataset(Dataset):
    def __init__(self, img_dir, coord_dir, img_size=512, sigma=4.0, is_train=True):
        self.img_dir = img_dir
        self.coord_dir = coord_dir
        self.img_size = img_size
        self.sigma = sigma  # sigma 用 4.0 比较适中
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        
        if is_train:
            # 只做像素级增强，绝不做旋转、平移！
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.2), 
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        coord_path = os.path.join(self.coord_dir, fname.replace('.npy', '_coords.npy'))

        img = np.load(img_path).astype(np.float32)
        coords = np.load(coord_path)
        
        # 清理异常坐标（如果你之前的检查发现有点越界，这里强制拉回）
        coords = np.clip(coords, 0, self.img_size - 1)
        coords = np.nan_to_num(coords, nan=0.0)

        # 归一化
        img = img / (np.max(img) + 1e-8)

        # 数据增强（只处理图像，不处理坐标）
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        # 生成 36 通道高斯热图
        heatmap = np.zeros((36, self.img_size, self.img_size), dtype=np.float32)
        yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
        
        for ch in range(36):
            x, y = coords[ch]
            heatmap[ch] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        heatmap_tensor = torch.from_numpy(heatmap)

        return img_tensor, heatmap_tensor